"""Tests for VRD ↔ COCO label mapping utilities."""

import torch

from src.data.label_mapping import VRDCOCOMapper


class TestVRDCOCOMapper:
    """Test suite for VRDCOCOMapper."""

    def test_init_loads_mappings(self):
        """Mapper initializes and loads VRD and COCO class names."""
        mapper = VRDCOCOMapper()

        assert mapper.vrd_classes is not None
        assert mapper.coco_classes is not None
        assert len(mapper.vrd_classes) == 100
        assert len(mapper.coco_classes) == 80

    def test_get_shared_classes_returns_overlap(self):
        """get_shared_classes returns classes present in both datasets."""
        mapper = VRDCOCOMapper()

        shared = mapper.get_shared_classes()

        # Should be a list of class names
        assert isinstance(shared, list)
        assert len(shared) > 0

        # Each shared class should be in VRD classes
        for class_name in shared:
            assert class_name in mapper.vrd_classes

            # The normalized version should match between VRD and COCO
            vrd_idx = mapper.vrd_classes.index(class_name)
            # If it's shared, it should have a valid mapping to COCO
            assert vrd_idx in mapper._vrd_to_coco_map

    def test_vrd_to_coco_maps_shared_classes(self):
        """vrd_to_coco maps VRD labels to COCO labels for shared classes."""
        mapper = VRDCOCOMapper()

        # Find a shared class (e.g., "person")
        if "person" in mapper.get_shared_classes():
            vrd_idx = mapper.vrd_classes.index("person")
            vrd_labels = torch.tensor([vrd_idx])

            coco_labels = mapper.vrd_to_coco(vrd_labels)

            # COCO person class is ID 0 (1-indexed becomes 1 in COCO API)
            assert coco_labels.shape == vrd_labels.shape
            assert coco_labels[0] >= 0  # Valid COCO label

    def test_vrd_to_coco_unmapped_classes_become_negative_one(self):
        """vrd_to_coco maps unmapped VRD classes to -1."""
        mapper = VRDCOCOMapper()

        # Find a VRD-only class
        vrd_only = [c for c in mapper.vrd_classes if c not in mapper.coco_classes]
        if vrd_only:
            vrd_idx = mapper.vrd_classes.index(vrd_only[0])
            vrd_labels = torch.tensor([vrd_idx])

            coco_labels = mapper.vrd_to_coco(vrd_labels)

            assert coco_labels[0] == -1

    def test_vrd_to_coco_handles_batches(self):
        """vrd_to_coco handles batch of labels."""
        mapper = VRDCOCOMapper()

        vrd_labels = torch.tensor([0, 1, 2, 3, 4])
        coco_labels = mapper.vrd_to_coco(vrd_labels)

        assert coco_labels.shape == vrd_labels.shape
        assert coco_labels.dtype == vrd_labels.dtype

    def test_coco_to_vrd_maps_shared_classes(self):
        """coco_to_vrd maps COCO labels to VRD labels for shared classes."""
        mapper = VRDCOCOMapper()

        # COCO person class is ID 0 (in 0-indexed format)
        if "person" in mapper.get_shared_classes():
            coco_idx = mapper.coco_classes.index("person")
            coco_labels = torch.tensor([coco_idx])

            vrd_labels = mapper.coco_to_vrd(coco_labels)

            assert vrd_labels.shape == coco_labels.shape
            assert vrd_labels[0] >= 0  # Valid VRD label

    def test_coco_to_vrd_unmapped_classes_become_negative_one(self):
        """coco_to_vrd maps unmapped COCO classes to -1."""
        mapper = VRDCOCOMapper()

        # Find a COCO-only class
        coco_only = [c for c in mapper.coco_classes if c not in mapper.vrd_classes]
        if coco_only:
            coco_idx = mapper.coco_classes.index(coco_only[0])
            coco_labels = torch.tensor([coco_idx])

            vrd_labels = mapper.coco_to_vrd(coco_labels)

            assert vrd_labels[0] == -1

    def test_filter_to_shared_removes_unmapped_classes(self):
        """filter_to_shared removes detections with unmapped classes."""
        mapper = VRDCOCOMapper()

        # Create mock detections with mix of mapped and unmapped classes
        boxes = torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [5.0, 5.0, 15.0, 15.0],
                [10.0, 10.0, 20.0, 20.0],
            ]
        )
        labels = torch.tensor([0, 1, 2])  # Some may be unmapped
        scores = torch.tensor([0.9, 0.8, 0.7])

        filtered_boxes, filtered_labels, filtered_scores = mapper.filter_to_shared(
            boxes, labels, scores
        )

        # Check shapes are consistent
        assert filtered_boxes.shape[0] == filtered_labels.shape[0]
        assert filtered_boxes.shape[0] == filtered_scores.shape[0]

        # Check we only kept valid mappings (no -1 labels)
        assert all(filtered_labels >= 0)

    def test_filter_to_shared_empty_input(self):
        """filter_to_shared handles empty inputs."""
        mapper = VRDCOCOMapper()

        boxes = torch.empty(0, 4)
        labels = torch.empty(0, dtype=torch.long)
        scores = torch.empty(0)

        filtered_boxes, filtered_labels, filtered_scores = mapper.filter_to_shared(
            boxes, labels, scores
        )

        assert filtered_boxes.shape == (0, 4)
        assert filtered_labels.shape == (0,)
        assert filtered_scores.shape == (0,)

    def test_filter_to_shared_all_mapped(self):
        """filter_to_shared preserves all detections if all classes are mapped."""
        mapper = VRDCOCOMapper()

        # Use only shared class labels
        shared_classes = mapper.get_shared_classes()
        if len(shared_classes) > 0:
            # Get VRD indices for shared classes
            shared_vrd_indices = [
                mapper.vrd_classes.index(c) for c in shared_classes[:3]
            ]

            boxes = torch.tensor(
                [
                    [0.0, 0.0, 10.0, 10.0],
                    [5.0, 5.0, 15.0, 15.0],
                    [10.0, 10.0, 20.0, 20.0],
                ]
            )
            labels = torch.tensor(shared_vrd_indices)
            scores = torch.tensor([0.9, 0.8, 0.7])

            filtered_boxes, filtered_labels, filtered_scores = mapper.filter_to_shared(
                boxes, labels, scores
            )

            # All should be preserved
            assert filtered_boxes.shape[0] == boxes.shape[0]
            assert filtered_labels.shape[0] == labels.shape[0]
            assert filtered_scores.shape[0] == scores.shape[0]

    def test_vrd_to_coco_background_class(self):
        """vrd_to_coco handles background class (if using 1-indexed)."""
        mapper = VRDCOCOMapper()

        # If using 1-indexed labels (0 = background)
        labels = torch.tensor([0])  # Background
        mapped = mapper.vrd_to_coco(labels)

        # Background should map to background or be handled consistently
        assert mapped.shape == labels.shape

    def test_roundtrip_consistency_for_shared_classes(self):
        """VRD→COCO→VRD roundtrip preserves shared class labels."""
        mapper = VRDCOCOMapper()

        shared_classes = mapper.get_shared_classes()
        if len(shared_classes) > 0:
            # Pick a shared class
            class_name = shared_classes[0]
            vrd_idx = mapper.vrd_classes.index(class_name)

            vrd_label = torch.tensor([vrd_idx])
            coco_label = mapper.vrd_to_coco(vrd_label)
            roundtrip_label = mapper.coco_to_vrd(coco_label)

            # Should get back the original VRD label
            if coco_label[0] != -1:  # If it mapped successfully
                assert roundtrip_label[0] == vrd_label[0]
