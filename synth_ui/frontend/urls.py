"""URL patterns for the frontend app, mapping views to routes for upload, edit, generate, and download operations."""

from django.urls import path

from .views import save_table_changes  # Corrected the function name
from .views import (
    download_synthetic_csv,
    edit_table,
    evaluation_status,
    generate_synthetic,
    start_evaluation,
    upload_csv,
    upload_page,
)

urlpatterns = [
    path("", upload_page, name="upload_page"),
    path("upload/", upload_csv, name="upload_csv"),
    path("table/<int:table_id>/", edit_table, name="edit_table"),
    path("table/<int:table_id>/save/", save_table_changes, name="save_table"),
    path(
        "table/<int:table_id>/synthesize/",
        generate_synthetic,
        name="generate_synthetic",
    ),
    path(
        "table/<int:table_id>/download/",
        download_synthetic_csv,
        name="download_synthetic",
    ),
    path(
        "table/<int:table_id>/evaluate/start/",
        start_evaluation,
        name="start_evaluation",
    ),
    path(
        "table/<int:table_id>/evaluate/<uuid:eval_id>/status/",
        evaluation_status,
        name="evaluation_status",
    ),
]
