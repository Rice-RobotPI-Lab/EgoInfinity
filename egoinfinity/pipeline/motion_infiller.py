"""
HaWoR motion infiller integration for EgoInfinity pipeline.

Fills missing hand pose frames using a pretrained Transformer model
from HaWoR. Operates in camera space (static camera assumed).

Usage in pipeline:
    infiller = MotionInfiller(checkpoint_path, device='cuda')
    infiller.fill_missing_frames(hand_results_per_frame, total_frames)
"""
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from .hand_reconstructor import HandResult
from .infiller_utils.network import TransformerModel
from .infiller_utils.filling import (
    filling_preprocess, filling_postprocess,
    slerp_interpolation_aa, linear_interpolation_nd,
)
from .infiller_utils.rotation import (
    angle_axis_to_rotation_matrix,
    rotation_matrix_to_angle_axis,
)

# Infiller network hyperparameters (must match HaWoR pretrained weights)
_POS_DIM = 3
_SHAPE_DIM = 10
_NUM_JOINTS = 15
_ROT_DIM = (_NUM_JOINTS + 1) * 6  # 96 (rot6d)
_REPR_DIM = 2 * (_POS_DIM + _SHAPE_DIM + _ROT_DIM)  # 218
_HORIZON = 120
_D_MODEL = 384
_NHEAD = 8
_D_HID = 2048
_NLAYERS = 8
_DROPOUT = 0.05

# Minimum ratio of both-hands-valid frames for Transformer infilling
_MIN_DUAL_VALID_RATIO = 0.15


def _rotmat_to_aa(rotmat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix (..., 3, 3) to angle-axis (..., 3) via torch."""
    shape = rotmat.shape[:-2]
    flat = torch.from_numpy(rotmat.reshape(-1, 3, 3)).float()
    aa = rotation_matrix_to_angle_axis(flat)
    return aa.numpy().reshape(*shape, 3)


class MotionInfiller:
    """Wraps the HaWoR TransformerModel for motion gap-filling."""

    def __init__(self, checkpoint_path: str, device: str = 'cuda',
                 mano_model=None):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model = TransformerModel(
            seq_len=_HORIZON, input_dim=_REPR_DIM, d_model=_D_MODEL,
            nhead=_NHEAD, d_hid=_D_HID, nlayers=_NLAYERS,
            dropout=_DROPOUT, out_dim=_REPR_DIM, masked_attention_stage=True,
        )
        self.model.to(self.device)
        self.model.load_state_dict(ckpt['transformer_encoder_state_dict'])
        self.model.eval()

        # MANO model for recomputing joints/vertices on filled frames
        self.mano = mano_model

    # ── Public API ────────────────────────────────────────────────────

    def fill_missing_frames(
        self,
        hand_results_per_frame: List[List[HandResult]],
        total_frames: int,
        aligned_cam_ts: Optional[List[List]] = None,
    ) -> List[List[HandResult]]:
        """Fill missing hand detections in-place using the infiller network.

        For frames where a hand was detected, MANO parameters are kept unchanged.
        Only missing frames are filled.  The method modifies HandResult lists
        and returns the updated version.

        Args:
            hand_results_per_frame: list of length total_frames, each element
                is a list of HandResult for that frame.
            total_frames: number of frames in the video.
            aligned_cam_ts: optional depth-aligned cam_t per frame per hand.
                If provided, used instead of h.cam_t for infiller input.

        Returns:
            Updated hand_results_per_frame with filled frames appended.
        """
        # Group detections by handedness (left=0, right=1)
        left_by_frame: Dict[int, HandResult] = {}
        right_by_frame: Dict[int, HandResult] = {}

        # Build a per-frame, per-hand aligned cam_t lookup
        aligned_lookup = {}  # (frame_idx, is_right) -> cam_t
        for fi in range(total_frames):
            for hi, h in enumerate(hand_results_per_frame[fi]):
                # Use aligned cam_t if available, else original
                if aligned_cam_ts is not None and fi < len(aligned_cam_ts) \
                        and hi < len(aligned_cam_ts[fi]) \
                        and aligned_cam_ts[fi][hi] is not None:
                    aligned_lookup[(fi, h.is_right)] = aligned_cam_ts[fi][hi]
                else:
                    aligned_lookup[(fi, h.is_right)] = h.cam_t

        for fi in range(total_frames):
            for h in hand_results_per_frame[fi]:
                if h.is_right:
                    right_by_frame[fi] = h
                else:
                    left_by_frame[fi] = h

        n_left = len(left_by_frame)
        n_right = len(right_by_frame)

        # Nothing to fill
        if n_left == 0 and n_right == 0:
            return hand_results_per_frame

        # Check if enough dual-hand frames for Transformer infilling
        both_valid = sum(1 for fi in range(total_frames)
                         if fi in left_by_frame and fi in right_by_frame)
        total_valid = sum(1 for fi in range(total_frames)
                          if fi in left_by_frame or fi in right_by_frame)

        if total_valid == 0:
            return hand_results_per_frame

        dual_ratio = both_valid / max(total_valid, 1)
        use_transformer = dual_ratio >= _MIN_DUAL_VALID_RATIO

        if use_transformer and both_valid >= 2:
            print(f"[Infiller] Using Transformer (dual={both_valid}/{total_valid}={dual_ratio:.0%})")
            filled = self._transformer_fill(
                left_by_frame, right_by_frame, total_frames, aligned_lookup)
        else:
            print(f"[Infiller] Using SLERP fallback (dual={both_valid}/{total_valid}={dual_ratio:.0%})")
            filled = self._slerp_fill(
                left_by_frame, right_by_frame, total_frames, aligned_lookup)

        if filled is None:
            return hand_results_per_frame

        # Build valid mask for injection (avoid zero-check heuristic)
        valid_mask = np.zeros((2, total_frames), dtype=bool)
        for fi in left_by_frame:
            valid_mask[0, fi] = True
        for fi in right_by_frame:
            valid_mask[1, fi] = True

        # Inject filled frames back
        self._inject_filled(
            hand_results_per_frame, filled, valid_mask,
            left_by_frame, right_by_frame,
            total_frames, aligned_lookup)

        return hand_results_per_frame

    # ── Transformer-based filling ────────────────────────────────────

    def _transformer_fill(
        self,
        left_by_frame: Dict[int, HandResult],
        right_by_frame: Dict[int, HandResult],
        total_frames: int,
        aligned_lookup: Dict = None,
    ) -> Optional[dict]:
        """Run the Transformer infiller on the full sequence."""

        # Build (2, T, ...) arrays from HandResult dicts
        trans = np.zeros((2, total_frames, 3), dtype=np.float32)
        rot = np.zeros((2, total_frames, 3), dtype=np.float32)
        hand_pose = np.zeros((2, total_frames, 45), dtype=np.float32)
        betas_arr = np.zeros((2, total_frames, 10), dtype=np.float32)
        valid = np.zeros((2, total_frames), dtype=bool)

        for fi, h in left_by_frame.items():
            rot_aa = _rotmat_to_aa(h.global_orient)
            pose_aa = _rotmat_to_aa(h.hand_pose)
            trans[0, fi] = aligned_lookup.get((fi, False), h.cam_t) if aligned_lookup else h.cam_t
            rot[0, fi] = rot_aa.flatten()[:3]
            hand_pose[0, fi] = pose_aa.flatten()
            betas_arr[0, fi] = h.betas
            valid[0, fi] = True

        for fi, h in right_by_frame.items():
            rot_aa = _rotmat_to_aa(h.global_orient)
            pose_aa = _rotmat_to_aa(h.hand_pose)
            trans[1, fi] = aligned_lookup.get((fi, True), h.cam_t) if aligned_lookup else h.cam_t
            rot[1, fi] = rot_aa.flatten()[:3]
            hand_pose[1, fi] = pose_aa.flatten()
            betas_arr[1, fi] = h.betas
            valid[1, fi] = True

        # Find the active range (first valid → last valid for either hand)
        any_valid = valid[0] | valid[1]
        valid_indices = np.where(any_valid)[0]
        if len(valid_indices) < 2:
            return None
        range_start = int(valid_indices[0])
        range_end = int(valid_indices[-1]) + 1

        results = {
            'trans': trans.copy(), 'rot': rot.copy(),
            'hand_pose': hand_pose.copy(), 'betas': betas_arr.copy(),
        }

        window_start = range_start
        while window_start < range_end:
            window_end = min(window_start + _HORIZON, range_end)
            seq_valid = valid[:, window_start:window_end]

            # Skip if no missing frames in this window
            if seq_valid.all():
                window_start += _HORIZON - 20  # overlap
                continue

            filling_seq = {
                'trans': trans[:, window_start:window_end].copy(),
                'rot': rot[:, window_start:window_end].copy(),
                'hand_pose': hand_pose[:, window_start:window_end].copy(),
                'betas': betas_arr[:, window_start:window_end].copy(),
                'valid': seq_valid,
            }

            filled_window = self._run_infiller(filling_seq)
            if filled_window is not None:
                # Only replace missing frames
                missing = ~seq_valid
                # TODO(B22): the next window will overwrite (not blend) any
                # missing frame in the 20-frame overlap region.  In the next
                # window those frames sit at the leftmost edge with only
                # right-side context, so their predictions are weaker than
                # the predictions we just wrote (which had bilateral context).
                # On long sequences (>120 frames) this can show up as a faint
                # seam at window boundaries.  Low-impact; track if we ever
                # see real artefacts.
                for key in ['trans', 'rot', 'hand_pose', 'betas']:
                    results[key][:, window_start:window_end][missing] = \
                        filled_window[key][missing]

                # Anchor filled trans to detected frames at gap boundaries
                # (preserves infiller's relative motion, fixes absolute depth)
                self._anchor_trans_to_boundaries(
                    results['trans'], valid, window_start, window_end)

            window_start += _HORIZON - 20  # 20-frame overlap for smooth blending

        return results

    @staticmethod
    def _anchor_trans_to_boundaries(trans, valid, window_start, window_end):
        """Anchor infiller's trans to detected frames at gap boundaries.

        Computes per-endpoint offset and linearly interpolates across the gap:
          offset_start = detected_prev - infiller[gap_start]
          offset_end   = detected_next - infiller[gap_end-1]
          infiller[t] += lerp(offset_start, offset_end, t/gap_len)

        Both ends align exactly (zero jump), infiller's motion shape preserved
        (only a linear ramp is added, curvature/acceleration unchanged).
        """
        T_total = valid.shape[1]
        for hand_idx in range(2):
            v = valid[hand_idx]
            t = window_start
            while t < window_end:
                if v[t]:
                    t += 1
                    continue
                gap_start = t
                while t < window_end and not v[t]:
                    t += 1
                gap_end = t

                prev_gi = None
                for s in range(gap_start - 1, -1, -1):
                    if v[s]:
                        prev_gi = s
                        break
                next_gi = None
                for s in range(gap_end, T_total):
                    if v[s]:
                        next_gi = s
                        break

                if prev_gi is None and next_gi is None:
                    continue

                gap_len = gap_end - gap_start

                # Compute offset at each end
                if prev_gi is not None:
                    offset_start = trans[hand_idx, prev_gi] - trans[hand_idx, gap_start]
                else:
                    offset_start = None

                if next_gi is not None:
                    offset_end = trans[hand_idx, next_gi] - trans[hand_idx, gap_end - 1]
                else:
                    offset_end = None

                # Apply linearly interpolated offset
                for k in range(gap_len):
                    gi = gap_start + k
                    if offset_start is not None and offset_end is not None:
                        w = k / max(gap_len - 1, 1)
                        offset = (1 - w) * offset_start + w * offset_end
                    elif offset_start is not None:
                        offset = offset_start
                    else:
                        offset = offset_end
                    trans[hand_idx, gi] += offset

    @torch.no_grad()
    def _run_infiller(self, filling_seq: dict) -> Optional[dict]:
        """Run one infiller forward pass on a sequence window.

        Args:
            filling_seq: dict with trans/rot/hand_pose/betas/valid arrays
                         for a window of frames.
        Returns:
            dict with filled trans/rot/hand_pose/betas, or None on failure.
        """
        seq_valid = filling_seq['valid']  # (2, T_window)

        try:
            filling_input, transform_info = filling_preprocess(filling_seq, mano=self.mano)
        except Exception:
            return None

        T_original = filling_input.shape[0]
        filling_length = _HORIZON

        # Convert to tensor: (T, 1, 218)
        inp = torch.from_numpy(filling_input).float().unsqueeze(1).to(self.device)
        # inp shape: (T, B=1, repr_dim)

        # Pad to _HORIZON if needed
        if T_original < filling_length:
            pad_len = filling_length - T_original
            padding = inp[-1:].repeat(pad_len, 1, 1)
            inp = torch.cat([inp, padding], dim=0)
            # Extend validity with True (padded = copy of last frame)
            pad_valid = np.ones((2, pad_len), dtype=bool)
            seq_valid_padded = np.concatenate([seq_valid, pad_valid], axis=1)
        else:
            seq_valid_padded = seq_valid

        T, B, _ = inp.shape

        # Build masks
        # valid = True where BOTH hands have real data
        both_valid = torch.from_numpy(
            seq_valid_padded[0] & seq_valid_padded[1]
        ).unsqueeze(1)  # (T, 1)

        # data_mask: (horizon, B, 1) — 1 for valid frames
        data_mask = torch.zeros((_HORIZON, B, 1), device=self.device, dtype=inp.dtype)
        data_mask[both_valid.expand(-1, B), :] = 1.0

        # attention_mask: (B, 1, T, T) — True = mask out (don't attend)
        atten_valid = both_valid.permute(1, 0).unsqueeze(1)  # (B, 1, T)
        atten_mask = torch.ones((B, 1, _HORIZON), device=self.device, dtype=torch.bool)
        atten_mask[:, :, :T] = ~atten_valid
        atten_mask = atten_mask.unsqueeze(2).expand(-1, -1, _HORIZON, -1)  # (B, 1, H, H)

        src_mask = torch.zeros(
            (filling_length, filling_length), device=self.device, dtype=torch.bool)

        output = self.model(inp, src_mask, data_mask, atten_mask)

        # Reshape: (T, B, 218) → (T, 2, 109)
        output = output[:T_original]
        output = output.squeeze(1).reshape(T_original, 2, -1).cpu().detach()

        return filling_postprocess(output, transform_info)

    # ── Fallback: simple SLERP + linear interpolation ────────────────

    def _slerp_fill(
        self,
        left_by_frame: Dict[int, HandResult],
        right_by_frame: Dict[int, HandResult],
        total_frames: int,
        aligned_lookup: Dict = None,
    ) -> Optional[dict]:
        """Simple interpolation fallback when dual-hand frames are scarce."""

        trans = np.zeros((2, total_frames, 3), dtype=np.float32)
        rot = np.zeros((2, total_frames, 1, 3), dtype=np.float32)
        hand_pose = np.zeros((2, total_frames, 15, 3), dtype=np.float32)
        betas_arr = np.zeros((2, total_frames, 10), dtype=np.float32)
        valid = np.zeros((2, total_frames), dtype=bool)

        for fi, h in left_by_frame.items():
            rot_aa = _rotmat_to_aa(h.global_orient)
            pose_aa = _rotmat_to_aa(h.hand_pose)
            trans[0, fi] = aligned_lookup.get((fi, False), h.cam_t) if aligned_lookup else h.cam_t
            rot[0, fi, 0] = rot_aa.flatten()[:3]
            hand_pose[0, fi] = pose_aa.reshape(15, 3)
            betas_arr[0, fi] = h.betas
            valid[0, fi] = True

        for fi, h in right_by_frame.items():
            rot_aa = _rotmat_to_aa(h.global_orient)
            pose_aa = _rotmat_to_aa(h.hand_pose)
            trans[1, fi] = aligned_lookup.get((fi, True), h.cam_t) if aligned_lookup else h.cam_t
            rot[1, fi, 0] = rot_aa.flatten()[:3]
            hand_pose[1, fi] = pose_aa.reshape(15, 3)
            betas_arr[1, fi] = h.betas
            valid[1, fi] = True

        # Interpolate per-hand independently
        trans_filled = linear_interpolation_nd(trans, valid)
        betas_filled = linear_interpolation_nd(betas_arr, valid)
        rot_filled = slerp_interpolation_aa(rot, valid)
        hand_pose_filled = slerp_interpolation_aa(hand_pose, valid)

        return {
            'trans': trans_filled,
            'rot': rot_filled.reshape(2, total_frames, 3),
            'hand_pose': hand_pose_filled.reshape(2, total_frames, 45),
            'betas': betas_filled,
        }

    # ── Inject filled results back into HandResult lists ─────────────

    @staticmethod
    def _find_gap_length(by_frame: Dict[int, 'HandResult'], fi: int) -> int:
        """Find the length of the gap containing frame fi."""
        frames = sorted(by_frame.keys())
        # Find prev and next valid frames
        prev_valid = None
        next_valid = None
        for f in reversed(frames):
            if f < fi:
                prev_valid = f
                break
        for f in frames:
            if f > fi:
                next_valid = f
                break
        if prev_valid is None or next_valid is None:
            return 999  # edge gap, treat as long
        return next_valid - prev_valid - 1

    def _inject_filled(
        self,
        hand_results_per_frame: List[List[HandResult]],
        filled: dict,
        valid_mask: np.ndarray,
        left_by_frame: Dict[int, HandResult],
        right_by_frame: Dict[int, HandResult],
        total_frames: int,
        aligned_lookup: Dict = None,
    ):
        """Write filled MANO params back as HandResult objects for missing frames.

        For short gaps (≤ 6 frames), uses neighbor interpolation instead of
        Transformer output for more stable results.

        Each filled frame inherits the track_id of its temporally nearest
        detected hand of the same handedness — not a single global one — so
        short gaps inside one tracking segment don't get tagged with the id of
        a later, unrelated segment.
        """
        # Determine the range where filling is meaningful
        fill_range = {}
        for hand_idx, by_frame in enumerate([left_by_frame, right_by_frame]):
            if by_frame:
                frames = sorted(by_frame.keys())
                fill_range[hand_idx] = (frames[0], frames[-1])

        for fi in range(total_frames):
            # Left hand
            if fi not in left_by_frame and left_by_frame:
                if 0 in fill_range and fill_range[0][0] <= fi <= fill_range[0][1]:
                    track_id = self._nearest_track_id(left_by_frame, fi)
                    gap_len = self._find_gap_length(left_by_frame, fi)
                    if gap_len <= 2:
                        hr = self._interpolate_from_neighbors(
                            left_by_frame, fi, is_right=False,
                            track_id=track_id, aligned_lookup=aligned_lookup)
                    else:
                        hr = self._make_hand_result(
                            filled, hand_idx=0, frame_idx=fi,
                            is_right=False, track_id=track_id,
                            reference=self._find_nearest_ref(left_by_frame, fi))
                    if hr is not None:
                        hand_results_per_frame[fi].append(hr)

            # Right hand
            if fi not in right_by_frame and right_by_frame:
                if 1 in fill_range and fill_range[1][0] <= fi <= fill_range[1][1]:
                    track_id = self._nearest_track_id(right_by_frame, fi)
                    gap_len = self._find_gap_length(right_by_frame, fi)
                    if gap_len <= 2:
                        hr = self._interpolate_from_neighbors(
                            right_by_frame, fi, is_right=True,
                            track_id=track_id, aligned_lookup=aligned_lookup)
                    else:
                        hr = self._make_hand_result(
                            filled, hand_idx=1, frame_idx=fi,
                            is_right=True, track_id=track_id,
                            reference=self._find_nearest_ref(right_by_frame, fi),
                        )
                    if hr is not None:
                        hand_results_per_frame[fi].append(hr)

    @staticmethod
    def _nearest_track_id(by_frame: Dict[int, HandResult], fi: int) -> int:
        """Pick track_id of the temporally nearest detected hand."""
        if not by_frame:
            return -1
        import bisect
        keys = sorted(by_frame.keys())
        pos = bisect.bisect_left(keys, fi)
        if pos == 0:
            return by_frame[keys[0]].track_id
        if pos >= len(keys):
            return by_frame[keys[-1]].track_id
        prev_k, next_k = keys[pos - 1], keys[pos]
        if abs(fi - prev_k) <= abs(next_k - fi):
            return by_frame[prev_k].track_id
        return by_frame[next_k].track_id

    @staticmethod
    def _find_nearest_ref(
        by_frame: Dict[int, HandResult], fi: int,
    ) -> Optional[HandResult]:
        """Find nearest valid HandResult to use as template for metadata."""
        if not by_frame:
            return None
        keys = sorted(by_frame.keys())
        # Binary search for closest
        import bisect
        pos = bisect.bisect_left(keys, fi)
        candidates = []
        if pos < len(keys):
            candidates.append(keys[pos])
        if pos > 0:
            candidates.append(keys[pos - 1])
        closest = min(candidates, key=lambda k: abs(k - fi))
        return by_frame[closest]

    @staticmethod
    def _interpolate_from_neighbors(
        by_frame: Dict[int, HandResult],
        fi: int,
        is_right: bool,
        track_id: int,
        aligned_lookup: Dict = None,
    ) -> Optional[HandResult]:
        """For short gaps (≤ 6 frames), linearly interpolate from flanking valid frames.

        More stable than Transformer output for short dropouts. Interpolates
        cam_t in the depth-aligned space (matching what detected frames will
        contribute to track_cam_ts) so the filled frame doesn't sit in a
        different coordinate system from its neighbors.
        """
        import bisect
        keys = sorted(by_frame.keys())
        pos = bisect.bisect_left(keys, fi)
        if pos == 0 or pos >= len(keys):
            return None
        prev_fi, next_fi = keys[pos - 1], keys[pos]
        prev_h, next_h = by_frame[prev_fi], by_frame[next_fi]

        # Interpolation weight
        t = (fi - prev_fi) / max(next_fi - prev_fi, 1)

        if aligned_lookup is not None:
            prev_cam_t = aligned_lookup.get((prev_fi, is_right), prev_h.cam_t)
            next_cam_t = aligned_lookup.get((next_fi, is_right), next_h.cam_t)
        else:
            prev_cam_t, next_cam_t = prev_h.cam_t, next_h.cam_t
        cam_t = (1 - t) * prev_cam_t + t * next_cam_t
        joints_3d_rel = (1 - t) * prev_h.joints_3d_rel + t * next_h.joints_3d_rel
        joints_3d = joints_3d_rel + cam_t
        # Vertices shift from each neighbor's own cam_t into the new aligned cam_t
        vertices = (1 - t) * (prev_h.vertices - prev_h.cam_t) + \
                   t * (next_h.vertices - next_h.cam_t) + cam_t
        betas = (1 - t) * prev_h.betas + t * next_h.betas
        # Lerp rotation matrices (approximate but fine for 1-2 frame gaps)
        global_orient = (1 - t) * prev_h.global_orient + t * next_h.global_orient
        hand_pose = (1 - t) * prev_h.hand_pose + t * next_h.hand_pose

        # joints_2d on detected hands is in full-image pixel coords (= x/z*f + cx,
        # +cy from hand_reconstructor.py:158-159). Lerp directly between the two
        # flanking detected frames — preserves the +cx,+cy offset and is exactly
        # right for ≤2-frame gaps where motion is small.
        joints_2d = ((1 - t) * np.asarray(prev_h.joints_2d, dtype=np.float32)
                     + t * np.asarray(next_h.joints_2d, dtype=np.float32))
        scaled_focal = prev_h.scaled_focal

        return HandResult(
            global_orient=global_orient,
            hand_pose=hand_pose,
            betas=betas,
            joints_3d=joints_3d,
            joints_3d_rel=joints_3d_rel,
            vertices=vertices,
            cam_t=cam_t,
            joints_2d=joints_2d,
            scaled_focal=scaled_focal,
            is_right=is_right,
            bbox=prev_h.bbox.copy(),
            confidence=0.0,
            track_id=track_id,
        )

    def _make_hand_result(
        self,
        filled: dict, hand_idx: int, frame_idx: int,
        is_right: bool, track_id: int,
        reference: Optional[HandResult],
    ) -> Optional[HandResult]:
        """Construct a HandResult from filled MANO parameters.

        If a MANO model is available, runs forward pass to get correct
        joints and vertices. Otherwise falls back to reference copy.
        """
        if reference is None:
            return None

        cam_t = filled['trans'][hand_idx, frame_idx]            # (3,)
        rot_aa = filled['rot'][hand_idx, frame_idx]             # (3,)
        pose_aa = filled['hand_pose'][hand_idx, frame_idx]      # (45,)
        betas_val = filled['betas'][hand_idx, frame_idx]        # (10,)

        # Convert angle-axis back to rotation matrices
        rot_mat = angle_axis_to_rotation_matrix(
            torch.from_numpy(rot_aa).float().unsqueeze(0)
        ).numpy()  # (1, 3, 3)

        pose_mat = angle_axis_to_rotation_matrix(
            torch.from_numpy(pose_aa.reshape(15, 3)).float()
        ).numpy()  # (15, 3, 3)

        # Run MANO forward to get correct joints and vertices
        if self.mano is not None:
            joints_3d_rel, vertices_rel = self._run_mano_forward(
                rot_mat, pose_mat, betas_val, is_right)
        else:
            joints_3d_rel = reference.joints_3d_rel.copy()
            vertices_rel = reference.vertices - reference.cam_t

        joints_3d = joints_3d_rel + cam_t
        vertices = vertices_rel + cam_t

        # joints_2d projection needs the same +cx,+cy offset that
        # hand_reconstructor.py uses for detected frames (line 158-159).
        # Derive cx,cy from the reference HandResult: ref's joints_2d already
        # contains +cx,+cy, so reverse the projection on its valid joints and
        # take the median offset. Robust to one or two bad joints with z≤0.
        scaled_focal = reference.scaled_focal
        ref_j3d = reference.joints_3d  # (21,3) in camera frame, +cam_t applied
        ref_j2d = np.asarray(reference.joints_2d, dtype=np.float32)
        cx = cy = 0.0
        valid_ref = ref_j3d[:, 2] > 0.01
        if valid_ref.any():
            cx = float(np.median(ref_j2d[valid_ref, 0]
                                 - ref_j3d[valid_ref, 0] / ref_j3d[valid_ref, 2]
                                 * scaled_focal))
            cy = float(np.median(ref_j2d[valid_ref, 1]
                                 - ref_j3d[valid_ref, 1] / ref_j3d[valid_ref, 2]
                                 * scaled_focal))
        joints_2d = np.full((21, 2), np.nan, dtype=np.float32)
        valid = joints_3d[:, 2] > 0.01
        if valid.any():
            joints_2d[valid, 0] = (joints_3d[valid, 0] / joints_3d[valid, 2]
                                   * scaled_focal + cx)
            joints_2d[valid, 1] = (joints_3d[valid, 1] / joints_3d[valid, 2]
                                   * scaled_focal + cy)

        return HandResult(
            global_orient=rot_mat,
            hand_pose=pose_mat,
            betas=betas_val,
            joints_3d=joints_3d,
            joints_3d_rel=joints_3d_rel,
            vertices=vertices,
            cam_t=cam_t,
            joints_2d=joints_2d,
            scaled_focal=scaled_focal,
            is_right=is_right,
            bbox=reference.bbox.copy(),
            confidence=0.0,  # mark as filled, not detected
            track_id=track_id,
        )

    @torch.no_grad()
    def _run_mano_forward(
        self,
        global_orient: np.ndarray,
        hand_pose: np.ndarray,
        betas: np.ndarray,
        is_right: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run MANO forward pass to get joints and vertices.

        WiLoR's MANO is a right-hand model. For left hands, we flip x,
        run forward, and flip back (same approach as WiLoR).

        Args:
            global_orient: (1, 3, 3) rotation matrix
            hand_pose: (15, 3, 3) rotation matrices
            betas: (10,) shape parameters

        Returns:
            joints_3d_rel: (21, 3) root-relative joints
            vertices_rel: (778, 3) root-relative vertices
        """
        go = torch.from_numpy(global_orient).float().unsqueeze(0)  # (1, 1, 3, 3)
        hp = torch.from_numpy(hand_pose).float().unsqueeze(0)      # (1, 15, 3, 3)
        b = torch.from_numpy(betas).float().unsqueeze(0)           # (1, 10)

        mano_out = self.mano(global_orient=go, hand_pose=hp, betas=b, pose2rot=False)

        joints = mano_out.joints[0].numpy()     # (21, 3)
        vertices = mano_out.vertices[0].numpy()  # (778, 3)

        # WiLoR flips x-axis for left hands
        flip = (2 * int(is_right) - 1)
        joints[:, 0] *= flip
        vertices[:, 0] *= flip

        return joints, vertices
