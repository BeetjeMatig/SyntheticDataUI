"""Frontend view functions for uploading, editing, and generating synthetic data.

This module defines Django view handlers for:
    - Uploading CSV data
    - Editing uploaded tables
    - Invoking synthetic data generation
    - Downloading results as CSV

Functions:
    index_redirect: Redirect root URL to the upload page.
    upload_page: Render the CSV upload form.
    upload_csv: Handle CSV file uploads.
    edit_table: Render the table editing page.
    save_table_changes: Save changes to a table via Ajax.
    generate_synthetic: Generate synthetic data for a table.
    download_results: Download synthetic data as a CSV file.
"""

# Import necessary libraries
import csv
import json
import threading
from datetime import datetime
from io import StringIO, TextIOWrapper

import pandas as pd
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_GET, require_POST

from .evaluation import synth_and_evaluate
from .models import EvaluationResult, UploadedTable
from .synth import make_synthetic_housing


# --- Redirect '/' to '/upload/' ---
def index_redirect(request):
    """Redirect root URL to the upload page.

    Parameters
    ----------
    request : HttpRequest
        The incoming HTTP request.

    Returns
    -------
    HttpResponseRedirect
        Redirect response to the upload page.
    """
    return redirect("upload_page")


# --- Upload form view ---
def upload_page(request):
    """Render the CSV upload form.

    Parameters
    ----------
    request : HttpRequest
        The incoming HTTP request.

    Returns
    -------
    HttpResponse
        Rendered upload page template.
    """
    return render(request, "frontend/upload.html")


# --- Handle CSV upload ---
def upload_csv(request):
    """Handle CSV file upload and store it in UploadedTable model.

    Parameters
    ----------
    request : HttpRequest
        The incoming HTTP POST request with CSV file.

    Returns
    -------
    HttpResponseRedirect or HttpResponse
        Redirect to edit_table on success; render upload page with error on failure.
    """
    if request.method == "POST" and request.FILES.get("file"):
        file = request.FILES["file"]
        file_name = file.name.lower()

        # ✅ Block anything that doesn't end with .csv
        if not file_name.endswith(".csv"):
            return render(
                request,
                "frontend/upload.html",
                {"upload_error": "❌ Invalid file type. Only .csv files are allowed."},
            )

        # Proceed with decoding & parsing…
        try:
            decoded = TextIOWrapper(file, encoding="utf-8")
            reader = csv.reader(decoded)
            rows = list(reader)
        except UnicodeDecodeError:
            return render(
                request,
                "frontend/upload.html",
                {
                    "upload_error": "❌ Could not read file. Please upload a valid UTF-8 encoded CSV."
                },
            )

        if not rows:
            return render(
                request,
                "frontend/upload.html",
                {"upload_error": "❌ The CSV appears to be empty."},
            )

        columns, data = rows[0], rows[1:]
        table = UploadedTable.objects.last()
        if table:
            table.name = file.name
            table.columns = columns
            table.rows = data
            table.save()
        else:
            table = UploadedTable.objects.create(
                name=file.name,
                columns=columns,
                rows=data,
            )

        return redirect("edit_table", table_id=table.id)

    return redirect("upload_page")


# --- Table editing view ---
def edit_table(request, table_id):
    """Render the table editing interface for a specific UploadedTable.

    Parameters
    ----------
    request : HttpRequest
        The incoming HTTP request.
    table_id : int
        Primary key of the UploadedTable instance.

    Returns
    -------
    HttpResponse
        Rendered table editing template with current data.
    """
    table = get_object_or_404(UploadedTable, id=table_id)
    return render(
        request,
        "frontend/index.html",
        {
            "columns": table.columns,
            "rows": table.rows,
            "table_id": table.id,  # ✅ pass table_id to template
        },
    )


# --- Save table changes via Ajax ---
@require_POST  # Ensure only POST requests are allowed
def save_table_changes(request, table_id):
    """Save edited table rows via AJAX POST.

    Parameters
    ----------
    request : HttpRequest
        The incoming HTTP POST request containing JSON rows data.
    table_id : int
        Primary key of the UploadedTable instance.

    Returns
    -------
    JsonResponse
        JSON status indicating success or error.
    """
    table = get_object_or_404(UploadedTable, id=table_id)

    if request.method == "POST":
        data = json.loads(request.body)
        table.rows = data.get("rows", [])
        table.save()
        return JsonResponse({"status": "success"})

    return JsonResponse({"error": "Invalid method"}, status=405)


# --- Generate synthetic data ---
def generate_synthetic(request, table_id):
    """Generate synthetic data for an uploaded table and render results.

    Parameters
    ----------
    request : HttpRequest
        The incoming HTTP request (GET renders form, POST generates data).
    table_id : int
        Primary key of the UploadedTable instance.

    Returns
    -------
    HttpResponse
        Rendered template for data generation or results page.
    """
    table = get_object_or_404(UploadedTable, id=table_id)
    if request.method == "POST":
        num = int(request.POST["num_records"])
        seed = int(request.POST.get("seed", 42))
        df_lookup = pd.DataFrame(table.rows, columns=table.columns)
        # generate synthetic dataframe
        df_synth = make_synthetic_housing(num, df_lookup, seed=seed)
        # prepare data for rendering
        columns = list(df_synth.columns)
        rows = df_synth.values.tolist()
        return render(
            request,
            "frontend/results.html",
            {
                "columns": columns,
                "rows": rows,
                "table_id": table.id,
                "num_records": num,
                "seed": seed,
            },
        )

    return render(request, "frontend/generate.html", {"table": table})


# --- Download results ---
@require_GET
def download_results(request, table_id):
    """Download synthetic data as a CSV file based on query parameters.

    Parameters
    ----------
    request : HttpRequest
        The incoming HTTP GET request with 'num_records' and 'seed' params.
    table_id : int
        Primary key of the UploadedTable instance.

    Returns
    -------
    HttpResponse
        CSV file attachment response.
    """
    # endpoint to download CSV of synthetic data using query params
    table = get_object_or_404(UploadedTable, id=table_id)
    num = int(request.GET.get("num_records", 100))
    seed = int(request.GET.get("seed", 42))
    df_lookup = pd.DataFrame(table.rows, columns=table.columns)
    # filter out any rows missing a street name
    df_lookup = df_lookup[
        df_lookup["STRAATNAAM"].notna()
        & df_lookup["STRAATNAAM"].astype(str).str.strip().ne("")
    ]
    df_synth = make_synthetic_housing(num, df_lookup, seed=seed)
    resp = HttpResponse(content_type="text/csv")
    resp["Content-Disposition"] = 'attachment; filename="synthetic.csv"'
    from io import StringIO

    csv_buffer = StringIO()
    df_synth.to_csv(csv_buffer, index=False)
    resp.write(csv_buffer.getvalue())
    return resp


@require_POST
def start_evaluation(request, table_id):
    """Start an asynchronous evaluation run."""
    data = json.loads(request.body)
    eval_type = data.get("type")
    num = int(data.get("num_records", 5000))
    seed = int(data.get("seed", 42))
    table = get_object_or_404(UploadedTable, id=table_id)
    # create EvaluationResult
    er = EvaluationResult.objects.create(
        table=table,
        evaluation_type=eval_type,
        status="pending",
    )

    def run_eval():
        er.status = "running"
        er.save()
        try:
            metrics = synth_and_evaluate(table, eval_type, num, seed)  # type: ignore
            er.metrics = metrics  # type: ignore
            er.status = "done"
        except Exception as e:
            er.status = "failed"
            er.error = str(e)  # type: ignore
        er.finished_at = datetime.utcnow()
        er.save()

    threading.Thread(target=run_eval, daemon=True).start()
    return JsonResponse({"id": str(er.id), "status": er.status})


@require_GET
def evaluation_status(request, table_id, eval_id):
    """Get status and results of an evaluation run."""
    er = get_object_or_404(EvaluationResult, id=eval_id, table__id=table_id)
    return JsonResponse(
        {
            "id": str(er.id),
            "type": er.evaluation_type,
            "status": er.status,
            "metrics": er.metrics,
            "error": er.error,
        }
    )


# --- Download synthetic data as CSV ---
@require_GET
def download_synthetic_csv(request, table_id):
    """Download synthetic data as a CSV file.

    Parameters
    ----------
    request : HttpRequest
        The incoming HTTP GET request with 'num_records' and 'seed' params.
    table_id : int
        Primary key of the UploadedTable instance.

    Returns
    -------
    HttpResponse
        CSV file attachment response.
    """
    table = get_object_or_404(UploadedTable, id=table_id)
    num_records = int(request.GET.get("num_records", 100))
    seed = int(request.GET.get("seed", 42))

    # Generate synthetic data
    df_lookup = pd.DataFrame(table.rows, columns=table.columns)
    df_synth = make_synthetic_housing(num_records, df_lookup, seed=seed)

    # Write synthetic data to a CSV buffer
    csv_buffer = StringIO()
    df_synth.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)  # Move to the beginning of the buffer

    # Create CSV response
    response = HttpResponse(csv_buffer, content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="synthetic_data.csv"'
    return response
