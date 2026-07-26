let video = document.getElementById("video");

let canvas = document.getElementById("canvas");

let preview = document.getElementById("preview");

let captureBtn = document.getElementById("captureBtn");

let analyzeBtn = document.getElementById("analyzeBtn");

let switchBtn = document.getElementById("switchBtn");

let currentFacing = "environment";

let capturedImage = null;

// Open Camera

async function startCamera() {
  if (window.stream) {
    window.stream.getTracks().forEach((track) => track.stop());
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: currentFacing,
    },

    audio: false,
  });

  window.stream = stream;

  video.srcObject = stream;
}

startCamera();

// Switch Camera

switchBtn.onclick = () => {
  if (currentFacing === "environment") {
    currentFacing = "user";
  } else {
    currentFacing = "environment";
  }

  startCamera();
};

// Capture Image

captureBtn.onclick = () => {
  canvas.width = video.videoWidth;

  canvas.height = video.videoHeight;

  canvas.getContext("2d").drawImage(
    video,

    0,

    0,
  );

  capturedImage = canvas.toDataURL("image/png");

  preview.src = capturedImage;

  preview.style.display = "block";
};

// Analyze

analyzeBtn.onclick = async () => {
  if (capturedImage == null) {
    alert("Capture image first");

    return;
  }

  const response = await fetch("/capture", {
    method: "POST",

    headers: {
      "Content-Type": "application/json",
    },

    body: JSON.stringify({
      image: capturedImage,

      food_type: document.getElementById("food_type").value,

      bought_value: document.getElementById("bought_value").value,

      time_unit: document.getElementById("time_unit").value,
    }),
  });

  const data = await response.json();

  let result = document.getElementById("result");

  result.style.display = "block";

  result.innerHTML = `

    <h2>AI Analysis Result</h2>

    <hr><br>

    <b>Food Type :</b> ${data.food}<br><br>

    <b>Status :</b> ${data.status}<br><br>

    <b>Spoilage Level :</b> ${data.level}<br><br>

    <b>Estimated pH :</b> ${data.estimated_ph}<br><br>

    <b>Normal pH :</b> ${data.normal_ph}<br><br>

    <b>Shelf Life Remaining :</b> ${data.remaining_days} Days<br><br>

    <b>Storage Suggestion :</b> ${data.storage}<br><br>

    <b>Precaution :</b> ${data.precaution}<br><br>

    `;
};
