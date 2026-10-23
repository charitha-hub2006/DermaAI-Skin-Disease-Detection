/**
 * Skin Disease Detection - Client-side Interactive Script
 * Handles drag-and-drop file upload, instant image preview, validation, and loading spinners.
 */

document.addEventListener("DOMContentLoaded", () => {
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const previewContainer = document.getElementById("previewContainer");
  const previewImage = document.getElementById("previewImage");
  const fileNameDisplay = document.getElementById("fileNameDisplay");
  const submitBtn = document.getElementById("submitBtn");
  const uploadForm = document.getElementById("uploadForm");
  const loadingSpinner = document.getElementById("loadingSpinner");
  const errorMessage = document.getElementById("errorMessage");

  if (!dropZone || !fileInput) return;
  dropZone.addEventListener("click", (event) => {
  if (event.target !== fileInput) {
    fileInput.click();
  }
});

dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});
  const MAX_FILE_SIZE = 16 * 1024 * 1024; // 16 MB
  const ALLOWED_TYPES = ["image/jpeg", "image/jpg", "image/png"];

  function showError(msg) {
    if (errorMessage) {
      errorMessage.textContent = msg;
      errorMessage.classList.remove("d-none");
    } else {
      alert(msg);
    }
    if (submitBtn) submitBtn.disabled = true;
  }

  function clearError() {
    if (errorMessage) {
      errorMessage.textContent = "";
      errorMessage.classList.add("d-none");
    }
  }

  function handleFile(file) {
    clearError();

    if (!file) {
      showError("No file selected.");
      return;
    }

    if (!ALLOWED_TYPES.includes(file.type.toLowerCase())) {
      showError("Invalid file type. Please upload a valid JPG, JPEG, or PNG image.");
      fileInput.value = "";
      return;
    }

    if (file.size > MAX_FILE_SIZE) {
      showError("File size exceeds 16 MB limit. Please choose a smaller image.");
      fileInput.value = "";
      return;
    }

    // Read and display preview
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      previewContainer.classList.remove("d-none");
      if (fileNameDisplay) {
        fileNameDisplay.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
      }
      if (submitBtn) submitBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  // File input change
  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  });

  // Drag and drop events
  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      fileInput.files = e.dataTransfer.files;
      handleFile(e.dataTransfer.files[0]);
    }
  });

  // Form submit loading state
  if (uploadForm) {
    uploadForm.addEventListener("submit", () => {
      if (submitBtn) {
        submitBtn.disabled = true;
      }
      if (loadingSpinner) {
        loadingSpinner.classList.remove("d-none");
      }
    });
  }
});
