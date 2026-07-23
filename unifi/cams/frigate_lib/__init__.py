"""
Supporting modules for unifi.cams.frigate.FrigateCam:
frigate_zones, frigate_tracker_ids, frigate_snapshots, frigate_motion,
frigate_descriptors, frigate_events.

This package intentionally does not import or re-export FrigateCam itself
(that would create a circular import, since FrigateCam imports from these
submodules) -- import it from `unifi.cams.frigate` instead.
"""
