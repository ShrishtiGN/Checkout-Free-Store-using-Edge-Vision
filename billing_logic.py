import numpy as np

class VirtualContainmentZone:
    """
    Defines a 3D bounding region representing a shopper's 'float-bag' or personal space.
    Replaces previous hardware weight-scale logic.
    """
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max):
        self.bounds = (x_min, x_max, y_min, y_max, z_min, z_max)
        
    def contains(self, centroid_3d):
        """
        Check if a given 3D centroid is inside the virtual float-bag.
        centroid_3d: tuple of (x, y, z)
        """
        x, y, z = centroid_3d
        x_min, x_max, y_min, y_max, z_min, z_max = self.bounds
        return (x_min <= x <= x_max) and (y_min <= y <= y_max) and (z_min <= z <= z_max)


class CartManager:
    """
    Manages the virtual cart by relying strictly on spatial containment 
    and temporal consistency, completely eliminating all dependencies on weight-scale triggers.
    """
    def __init__(self, containment_zone: VirtualContainmentZone, required_frames=16):
        """
        required_frames: The number of CONSECUTIVE frames an item must stay 
                         within the containment zone to trigger 'Add to Cart'.
                         Defaults to 16 to satisfy the >15 frames condition.
        """
        self.containment_zone = containment_zone
        self.required_frames = required_frames
        
        # Track how many consecutive frames an item has been inside the containment zone
        self.item_frame_counts = {}
        
        # Track items already billed to prevent duplicate 'Add to Cart' events
        self.billed_items = set()
        
        # Note: Previous implementations relied on weight-scale sensors and hardware interrupt methods.
        # This completely rewrites that pipeline in favor of pure Computer Vision spatial dynamics.

    def update_cart_state(self, track_id, bbox_3d):
        """
        Checks an object's spatial position per frame and updates the billing logic.
        
        track_id: unique ID from the Re-ID Tracker
        bbox_3d: list or tuple of [x, y, z, w, h, d] bounding box from YOLO
        returns: Boolean indicating if an 'Add to Cart' transaction event occurred.
        """
        if track_id in self.billed_items:
            return False
            
        # Compute centroid from 3D bounding box (assuming [x,y,z] represent the volume center)
        x, y, z, w, h, d = bbox_3d
        centroid_3d = (x, y, z)
        
        if self.containment_zone.contains(centroid_3d):
            # Item centroid resides inside the float-bag boundaries
            self.item_frame_counts[track_id] = self.item_frame_counts.get(track_id, 0) + 1
            
            # Temporal Consistency Check (>15 frames)
            if self.item_frame_counts[track_id] >= self.required_frames:
                self._trigger_add_to_cart(track_id)
                return True
        else:
            # If the item drifts out before hitting the threshold, reset its temporal counter.
            # This perfectly isolates items casually floating past from those actually docked in a bag.
            if track_id in self.item_frame_counts:
                self.item_frame_counts[track_id] = 0
                
        return False

    def _trigger_add_to_cart(self, track_id):
        """
        Fires the transaction event.
        """
        # Execute the transaction payload
        print(f"[TRANSACTION EVENT] Item {track_id} remained in float-bag for >= {self.required_frames} frames.")
        print(f" -> ADDED TO VIRTUAL CART: {track_id}")
        
        # Mark as billed
        self.billed_items.add(track_id)
        
        # Flush from temporal memory to optimize loop
        if track_id in self.item_frame_counts:
            del self.item_frame_counts[track_id]
