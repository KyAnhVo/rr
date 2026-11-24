# Phases of the Project


## Idea
using SIFT key points + matching, given 2 keyframes that are "moving" and not pure rotation, we can back-project with them and construct those key points, and further, 
construct the point cloud of those key points.

## Algorithm

1. **Keyframe selection** - We pick keyframes from the input video that are as geometrically useful
as possible: we discard blurry frames (variance-of-Laplacian), and are sure to keep frames
that have enough parallax and feature-match overlap with the previous keyframe (so
avoiding the degenerate small-baseline situations between video frames).
2. **Sparse reconstruction** - Feature-based incremental SfM (COLMAP) with sequential
matching estimates camera intrinsics, camera poses, and produces a sparse 3D point
cloud.
3. **Dense reconstruction** - Multi-view stereo (PatchMatch) densifies the cloud. Low-texture
walls result in few reliable matches, so we also fit planar primitives (RANSAC) to recover
large structural surfaces.
4. **2D semantic segmentation** - Each keyframe is labeled by a 2D segmentation network
pretrained on indoor classes (ADE20K / NYUv2) collapsed to a binary structure vs. object
taxonomy (wall floor ceiling door window structure; everything else object).
5. **2D-3D label fusion** - Given the recovered poses and intrinsics, the perpixel labels are back-
projected to the 3D points and fused across views by multi-view voting, denoising single-
frame errors and that yields a final label for each 3D point.
