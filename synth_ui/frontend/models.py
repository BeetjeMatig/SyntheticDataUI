"""Database models for the frontend application, including storage of uploaded CSV tables.

This module defines the UploadedTable model used to persist CSV data and metadata.
"""

import uuid

from django.db import models


class UploadedTable(models.Model):
    """Model representing an uploaded CSV table with name, timestamp, columns, and rows."""

    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    columns = models.JSONField()
    rows = models.JSONField()

    def __str__(self):
        """Return the human-readable name of the uploaded table."""
        return self.name


class EvaluationResult(models.Model):
    """Store synthetic data evaluation runs and their results."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    table = models.ForeignKey(UploadedTable, on_delete=models.CASCADE)
    evaluation_type = models.CharField(
        max_length=50,
        choices=[
            ("descriptive", "Descriptive"),
            ("performance", "Performance"),
            ("privacy", "Privacy"),
        ],
    )
    status = models.CharField(
        max_length=20,
        choices=[
            ("pending", "Pending"),
            ("running", "Running"),
            ("done", "Done"),
            ("failed", "Failed"),
        ],
        default="pending",
    )
    metrics = models.JSONField(null=True, blank=True)
    error = models.TextField(null=True, blank=True)
    started_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"Evaluation {self.id} ({self.evaluation_type})"
