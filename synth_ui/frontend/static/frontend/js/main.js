/**
 * main.js - Table editing, pagination, and UI utilities for synthetic data frontend.
 *
 * Provides functions to toggle edit mode, save changes via AJAX, paginate table rows,
 * display toast messages, and handle column/row deletion confirmations.
 */

console.log("main.js script loaded.");

// Utility to get a cookie value (for CSRF token)
function getCookie(name) {
  let cookieValue = null;
  if (document.cookie && document.cookie !== "") {
    const cookies = document.cookie.split(";");
    for (let cookie of cookies) {
      cookie = cookie.trim();
      if (cookie.startsWith(name + "=")) {
        cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
        break;
      }
    }
  }
  return cookieValue;
}

let editMode = false;
let currentPage = 1;
const rowsPerPage = 10;
let columnToDeleteButton = null;

// Dynamically fetch tableId from the <body data-table-id="">
const tableId = document.body.dataset.tableId;

// Run after page loads
function init() {
  console.log("Initializing UI event listeners.");

  const editToggle = document.getElementById("edit-toggle");
  const saveButton = document.getElementById("save-button");
  const prevButton = document.getElementById("prev-btn");
  const nextButton = document.getElementById("next-btn");
  const fileUpload = document.getElementById("file-upload");

  if (editToggle) editToggle.addEventListener("click", toggleEditMode);
  if (saveButton) saveButton.addEventListener("click", saveChanges);
  if (prevButton) prevButton.addEventListener("click", prevPage);
  if (nextButton) nextButton.addEventListener("click", nextPage);
  if (fileUpload) fileUpload.addEventListener("change", showToast);

  // Auto-hide flash messages
  const messageContainer = document.querySelector(".fixed.top-6.right-6.z-50"); // Target the container
  if (messageContainer) {
    // Find all direct child divs that likely represent messages
    // Using role='alert' which was added to the template
    const messages = messageContainer.querySelectorAll(
      ":scope > div[role='alert']"
    );
    messages.forEach((message) => {
      // Ensure transition property is set for opacity changes
      message.style.transition = "opacity 0.5s ease-out";

      setTimeout(() => {
        message.style.opacity = "0";

        // Remove the element after the transition completes
        const handleTransitionEnd = () => {
          message.remove();
          message.removeEventListener("transitionend", handleTransitionEnd);
        };
        message.addEventListener("transitionend", handleTransitionEnd);

        // Fallback removal in case transitionend doesn't fire reliably
        setTimeout(() => {
          // Check if the element still exists before trying to remove
          if (message.parentNode) {
            message.remove();
          }
        }, 5600); // Slightly longer than the main timeout + transition duration
      }, 5000); // 5 seconds delay before starting fade
    });
  }

  renderPage();

  // Evaluation buttons
  document.querySelectorAll(".eval-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const type = btn.dataset.type;
      const feedback = document.getElementById("eval-feedback");
      const card = document.getElementById("evaluation-card");
      const numRecords = parseInt(card.dataset.numRecords, 10);
      const seed = parseInt(card.dataset.seed, 10);
      // disable buttons
      document
        .querySelectorAll(".eval-btn")
        .forEach((b) => (b.disabled = true));
      feedback.innerHTML = `<p>Starting ${type} evaluation...</p>`;
      const payload = {
        type: type,
        num_records: numRecords,
        seed: seed,
      };
      fetch(`/table/${tableId}/evaluate/start/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-CSRFToken": getCookie("csrftoken"),
        },
        body: JSON.stringify(payload),
      })
        .then((resp) => resp.json())
        .then((data) => {
          const evalId = data.id;
          feedback.innerHTML =
            "<p>Evaluation started (ID: " +
            evalId +
            "). Checking status...</p>";
          // poll status
          const interval = setInterval(() => {
            fetch(`/table/${tableId}/evaluate/${evalId}/status/`)
              .then((r) => r.json())
              .then((statusData) => {
                if (
                  statusData.status === "running" ||
                  statusData.status === "pending"
                ) {
                  feedback.innerHTML =
                    "<p>Evaluation " + statusData.status + "...</p>";
                } else {
                  clearInterval(interval);
                  if (statusData.status === "done") {
                    if (type === "descriptive") {
                      feedback.innerHTML = formatStatSummary(
                        statusData.metrics
                      );
+                      attachInfoIconListeners();
                    } else if (type === "performance") {
                      feedback.innerHTML = formatPerformanceMetrics(
                        statusData.metrics
                      );
+                      attachInfoIconListeners();
                    } else if (type === "privacy") {
                      feedback.innerHTML = formatPrivacyMetrics(
                        statusData.metrics
                      );
+                      attachInfoIconListeners();
                    } else {
                      feedback.innerHTML =
                        "<pre>" +
                        JSON.stringify(statusData.metrics, null, 2) +
                        "</pre>";
                    }
                  } else {
                    feedback.innerHTML =
                      '<p class="text-red-600">Error: ' +
                      statusData.error +
                      "</p>";
                  }
                  // re-enable buttons
                  document
                    .querySelectorAll(".eval-btn")
                    .forEach((b) => (b.disabled = false));
                }
              });
          }, 2000);
        })
        .catch((err) => {
          feedback.innerHTML =
            '<p class="text-red-600">Failed to start evaluation: ' +
            err +
            "</p>";
          document
            .querySelectorAll(".eval-btn")
            .forEach((b) => (b.disabled = false));
        });
    });
  });
}

document.addEventListener("DOMContentLoaded", init);

/**
 * Toggle inline edit mode for table cells.
 */
function toggleEditMode() {
  console.log("toggleEditMode called.");
  const toggle = document.getElementById("edit-toggle");
  const saveBtn = document.getElementById("save-button");
  const cells = document.querySelectorAll(
    "#data-table tbody td:not(:last-child)"
  );

  editMode = !editMode;

  cells.forEach((cell) => {
    if (editMode) {
      cell.setAttribute("contenteditable", "true");
      cell.classList.add("bg-yellow-50", "editing-cell");
    } else {
      cell.removeAttribute("contenteditable");
      cell.classList.remove("bg-yellow-50", "editing-cell");
    }
  });

  toggle.textContent = editMode ? "🛑 Exit Edit Mode" : "✏️ Edit";
  saveBtn.classList.toggle("hidden", !editMode);
}

/**
 * Collect edited table rows and send changes to server via POST.
 */
function saveChanges() {
  console.log("saveChanges called.");
  const rowsToSave = [];

  const headerCells = document.querySelectorAll("#data-table thead th");
  const expectedDataColumns = headerCells.length - 1; // Exclude action column

  document.querySelectorAll("#data-table tbody tr").forEach((tr) => {
    // Remove the check for tr.style.display to ensure all rows are saved
    // if (tr.style.display === "none") return;

    const cells = tr.querySelectorAll("td");
    if (cells.length >= expectedDataColumns) {
      const rowData = [];

      for (let i = 0; i < expectedDataColumns; i++) {
        let cellContent = cells[i].textContent.trim();
        const headerCell = headerCells[i];

        if (headerCell && headerCell.textContent.includes("Date")) {
          cellContent = parseDate(cellContent);
        }

        rowData.push(cellContent);
      }

      if (rowData.length === expectedDataColumns) {
        rowsToSave.push(rowData);
      }
    } else {
      console.warn("Skipping row due to unexpected cell count:", tr);
    }
  });

  fetch(`/table/${tableId}/save/`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-CSRFToken": getCookie("csrftoken"),
    },
    body: JSON.stringify({ rows: rowsToSave }),
  })
    .then((r) => {
      if (r.ok) {
        alert("Changes saved successfully!");
      } else {
        return r
          .json()
          .then((err) => {
            throw new Error(
              err.detail ||
                err.error ||
                `Server responded with status: ${r.status}`
            );
          })
          .catch(() => {
            throw new Error(
              `Failed to save changes. Server responded with status: ${r.status}`
            );
          });
      }
    })
    .catch((error) => {
      console.error("Save failed:", error);
      alert(`Failed to save changes: ${error.message}`);
    });
}

/**
 * Parse a date string and return ISO YYYY-MM-DD format if valid.
 * @param {string} text - The date string to parse.
 * @returns {string} Formatted date or original text if invalid.
 */
function parseDate(text) {
  const d = new Date(text);
  if (!isNaN(d)) {
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, "0");
    const day = String(d.getDate()).padStart(2, "0");
    return `${y}-${m}-${day}`;
  }
  return text;
}

/**
 * Delete a table row when delete button is clicked.
 * @param {HTMLElement} button - The delete button element.
 */
function deleteRow(button) {
  console.log("deleteRow called.");
  const row = button.closest("tr");
  row.remove();
  renderPage();
}

/**
 * Prompt user to confirm deletion of a column.
 * @param {HTMLElement} button - The delete column button element.
 */
function deleteColumn(button) {
  console.log("deleteColumn called.");
  columnToDeleteButton = button;
  showDeleteConfirmModal();
}

/**
 * Show the delete confirmation modal.
 */
function showDeleteConfirmModal() {
  document.getElementById("delete-confirm-modal").classList.remove("hidden");
  document.getElementById("main-content").classList.add("content-blur");
}

/**
 * Hide the delete confirmation modal.
 */
function hideDeleteConfirmModal() {
  document.getElementById("delete-confirm-modal").classList.add("hidden");
  document.getElementById("main-content").classList.remove("content-blur");
  columnToDeleteButton = null;
}

/**
 * After confirmation, remove the selected column from the table.
 */
function confirmDeleteColumnAction() {
  console.log("confirmDeleteColumnAction called.");
  if (!columnToDeleteButton) return;
  const th = columnToDeleteButton.closest("th");
  const idx = Array.from(th.parentNode.children).indexOf(th);
  document.querySelectorAll("#data-table tr").forEach((row) => {
    if (row.children[idx]) row.children[idx].remove();
  });
  hideDeleteConfirmModal();
  renderPage();
}

/**
 * Cancel column deletion and hide modal.
 */
function cancelDeleteColumnAction() {
  hideDeleteConfirmModal();
}

/**
 * Show a temporary toast notification after file upload.
 */
function showToast() {
  console.log("showToast called.");
  const t = document.getElementById("upload-toast");
  t.classList.remove("hidden");
  setTimeout(() => t.classList.add("hidden"), 5000);
}

/**
 * Close the upload toast notification.
 */
function closeToast() {
  document.getElementById("upload-toast").classList.add("hidden");
}

/**
 * Render a specific page of table rows and update navigation controls.
 */
function renderPage() {
  console.log(`renderPage called. Current page: ${currentPage}`);
  const rows = document.querySelectorAll("#data-table tbody tr");
  const totalRows = rows.length;
  const totalPages = Math.ceil(totalRows / rowsPerPage) || 1;

  currentPage = Math.min(Math.max(currentPage, 1), totalPages);

  const start = (currentPage - 1) * rowsPerPage;
  const end = start + rowsPerPage;

  rows.forEach((row, i) => {
    row.style.display = i >= start && i < end ? "" : "none";
  });

  const pageCounter = document.getElementById("page-counter");
  if (pageCounter) {
    pageCounter.textContent = `Page ${currentPage} of ${totalPages}`;
  }

  const prevBtn = document.getElementById("prev-btn");
  const nextBtn = document.getElementById("next-btn");

  if (prevBtn) prevBtn.disabled = currentPage === 1;
  if (nextBtn) nextBtn.disabled = currentPage === totalPages;
}

/**
 * Navigate to the next page of table rows.
 */
function nextPage() {
  console.log("nextPage called.");
  currentPage++;
  renderPage();
}

/**
 * Navigate to the previous page of table rows.
 */
function prevPage() {
  console.log("prevPage called.");
  currentPage--;
  renderPage();
}

// Format numerical summary metrics into an HTML table
function formatStatSummary(metrics) {
  // Explanations for descriptive statistics
  const statExplanations = {
    count: "Counts all non-missing records, showing how many valid data points are present.",
    mean: "Arithmetic average of the values, indicating the central tendency of the data.",
    std: "Standard deviation, measuring the dispersion of values around the mean.",
    min: "The smallest observed value in the dataset, indicating the lower bound.",
    "25%": "First quartile (25th percentile), below which 25% of the data points fall.",
    "50%": "Median (50th percentile), the midpoint that divides the data into two equal halves.",
    "75%": "Third quartile (75th percentile), below which 75% of data points fall.",
    max: "The largest observed value in the dataset, indicating the upper bound."
  };
   const ns = metrics.numeric_summary;
   const stats = Object.keys(ns);
   const variables = Object.keys(ns.count_real);
   let html =
     '<div class="overflow-x-auto"><table class="min-w-full table-auto border-collapse border border-gray-200">';
  // header
  html += '<thead><tr><th class="px-4 py-2 border border-gray-200"></th>';
  variables.forEach((variable) => {
    html += `<th class="px-4 py-2 border border-gray-200 bg-gray-50">${variable}</th>`;
  });
  html += "</tr></thead><tbody>";
  // rows
  stats.forEach((stat) => {
    // Prepare display name and tooltip
    const displayStat = stat.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
    const baseStat = stat.replace(/_real|_synth/g, "");
    const explanation = statExplanations[baseStat] || "";
    html += `<tr><td class="px-4 py-2 border border-gray-200 font-semibold bg-gray-100">${displayStat}<span class="info-icon ml-1 text-blue-500 cursor-pointer" data-tooltip="${explanation}" title="${explanation}">ℹ️</span></td>`;
     variables.forEach((variable) => {
       const value = ns[stat][variable];
       const display = typeof value === "number" ? value.toFixed(2) : value;
       html += `<td class="px-4 py-2 border border-gray-200 text-right">${display}</td>`;
     });
     html += "</tr>";
   });
   html += "</tbody></table></div>";
   return html;
}

// Format performance metrics into an HTML table
function formatPerformanceMetrics(metrics) {
    // Explanations for performance metrics
    const metricExplanations = {
        general_score: "General Score: a weighted aggregate summarizing all performance metrics into one overall quality indicator.",
        column_shapes: "Column Shapes: compares distribution shapes of each individual column between real and synthetic data.",
        pair_trends: "Pair Trends: evaluates how well relationships and correlations between column pairs are preserved.",
        category_coverage: "Category Coverage: fraction of real categorical levels that appear in the synthetic data to ensure diversity.",
        missing_value_similarity: "Missing Value Similarity: compares patterns and proportions of missing entries between datasets.",
        range_coverage: "Range Coverage: checks if synthetic values span the same numeric range as the real data.",
        statistic_similarity: "Statistic Similarity: measures how closely key statistics (mean, std) of numeric columns match."
    };
     let html =
         '<div class="overflow-x-auto"><table class="min-w-full table-auto border-collapse border border-gray-200">';
  // header
  html +=
    '<thead><tr><th class="px-4 py-2 border border-gray-200 bg-gray-50">Metric</th><th class="px-4 py-2 border border-gray-200 bg-gray-50">Value</th></tr></thead><tbody>';
  // rows
  Object.entries(metrics).forEach(([key, value]) => {
         const displayKey = key
             .replace(/_/g, " ")
             .replace(/\b\w/g, (c) => c.toUpperCase());
         const displayVal =
             typeof value === "number"
                 ? value.toFixed(2)
                 : value == null
                 ? "N/A"
                 : value;
        // Add info icon with tooltip
        const explanation = metricExplanations[key] || "";
        html += `<tr><td class="px-4 py-2 border border-gray-200 font-semibold bg-gray-100">${displayKey}<span class="info-icon ml-1 text-blue-500 cursor-pointer" data-tooltip="${explanation}" title="${explanation}">ℹ️</span></td><td class="px-4 py-2 border border-gray-200 text-right">${displayVal}</td></tr>`;
     });
     html += "</tbody></table></div>";
     return html;
}

// Format privacy metrics into an HTML table
function formatPrivacyMetrics(metrics) {
    // Explanations for privacy metrics
    const privacyExplanations = {
        dcr_overfitting: "Overfitting Protection: assesses if synthetic records are overly similar to training examples, indicating potential privacy risk.",
        dcr_baseline: "Baseline Protection: evaluates privacy risk by comparing synthetic data to a random baseline model.",
        disclosure: "Disclosure Protection: estimates how well sensitive attributes are shielded when subsets of known features are exposed.",
        privacy_score: "Privacy Score: a weighted combination of protection metrics to gauge overall privacy assurance."
    };
     let html =
         '<div class="overflow-x-auto"><table class="min-w-full table-auto border-collapse border border-gray-200">';
  html +=
    '<thead><tr><th class="px-4 py-2 border border-gray-200 bg-gray-50">Metric</th><th class="px-4 py-2 border border-gray-200 bg-gray-50">Value</th></tr></thead><tbody>';
  Object.entries(metrics).forEach(([key, value]) => {
         const displayKey = key
             .replace(/_/g, " ")
             .replace(/\b\w/g, (c) => c.toUpperCase());
         const displayVal =
             typeof value === "number"
                 ? value.toFixed(2)
                 : value == null
                 ? "N/A"
                 : value;
        // Add info icon with tooltip
        const explanation = privacyExplanations[key] || "";
        html += `<tr><td class="px-4 py-2 border border-gray-200 font-semibold bg-gray-100">${displayKey}<span class="info-icon ml-1 text-blue-500 cursor-pointer" data-tooltip="${explanation}" title="${explanation}">ℹ️</span></td><td class="px-4 py-2 border border-gray-200 text-right">${displayVal}</td></tr>`;
     });
     html += "</tbody></table></div>";
     return html;
}

// Attach click handlers to info icons for pop-up tooltips
function attachInfoIconListeners() {
  document.querySelectorAll('.info-icon').forEach(icon => {
    icon.addEventListener('click', () => {
      const msg = icon.getAttribute('data-tooltip') || '';
      alert(msg);
    });
  });
}
