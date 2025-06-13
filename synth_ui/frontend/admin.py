"""Admin site registration for frontend models."""

from django.contrib import admin

from .models import EvaluationResult, UploadedTable

# Register your models here.
admin.site.register(UploadedTable)
admin.site.register(EvaluationResult)
