const TRANSCRIBE_PATH = "/transcribe";
const FILE_FIELD = "file";
const RECORDING_FILENAME = "recording.webm";
const POLL_INTERVAL_MS = 3000;
const POLL_MAX_ATTEMPTS = 40;

const startButton = document.getElementById("start");
const stopButton = document.getElementById("stop");
const stateEl = document.getElementById("state");
const messageEl = document.getElementById("message");
const transcriptBox = document.getElementById("transcript-box");
const transcriptEl = document.getElementById("transcript");
const requestBox = document.getElementById("request-box");
const requestDebug = document.getElementById("request-debug");
const responseBox = document.getElementById("response-box");
const responseDebug = document.getElementById("response-debug");

let mediaRecorder = null;
let mediaStream = null;
let chunks = [];

function isLocalEnvironment() {
  const host = window.location.hostname;
  return host === "localhost" || host === "127.0.0.1" || host === "[::1]";
}

function setState(text, recording) {
  stateEl.textContent = text;
  stateEl.classList.toggle("recording", Boolean(recording));
}

function setMessage(text) {
  if (!text) {
    messageEl.hidden = true;
    messageEl.textContent = "";
    return;
  }
  messageEl.hidden = false;
  messageEl.textContent = text;
}

function resetDebug() {
  requestBox.hidden = true;
  responseBox.hidden = true;
  requestDebug.textContent = "";
  responseDebug.textContent = "";
  transcriptBox.hidden = true;
  transcriptEl.textContent = "";
}

function setIdleControls() {
  startButton.disabled = false;
  stopButton.disabled = true;
  setState("Idle", false);
}

function pickMimeType() {
  const candidates = ["audio/webm;codecs=opus", "audio/webm"];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

function stopTracks() {
  if (!mediaStream) {
    return;
  }
  for (const track of mediaStream.getTracks()) {
    track.stop();
  }
  mediaStream = null;
}

async function startRecording() {
  startButton.disabled = true;
  setMessage("");
  resetDebug();

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setMessage("Microphone capture is not available in this browser.");
    setIdleControls();
    return;
  }
  if (typeof MediaRecorder === "undefined") {
    setMessage("MediaRecorder is not available in this browser.");
    setIdleControls();
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (error) {
    const name = error && error.name;
    if (name === "NotAllowedError" || name === "PermissionDeniedError") {
      setMessage("Microphone permission was denied.");
    } else if (name === "NotFoundError" || name === "DevicesNotFoundError") {
      setMessage("No microphone was found.");
    } else {
      setMessage("Could not access the microphone.");
    }
    stopTracks();
    setIdleControls();
    return;
  }

  chunks = [];
  const mimeType = pickMimeType();
  try {
    mediaRecorder = mimeType
      ? new MediaRecorder(mediaStream, { mimeType })
      : new MediaRecorder(mediaStream);
  } catch (_error) {
    setMessage("Could not start MediaRecorder.");
    stopTracks();
    setIdleControls();
    return;
  }

  mediaRecorder.addEventListener("dataavailable", (event) => {
    if (event.data && event.data.size > 0) {
      chunks.push(event.data);
    }
  });
  mediaRecorder.addEventListener("stop", onRecorderStop);

  mediaRecorder.start();
  startButton.disabled = true;
  stopButton.disabled = false;
  setState("Recording…", true);
}

function stopRecording() {
  stopButton.disabled = true;
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
    return;
  }
  stopTracks();
  setIdleControls();
}

async function onRecorderStop() {
  const recorderType =
    (mediaRecorder && mediaRecorder.mimeType) || "audio/webm";
  mediaRecorder = null;
  stopTracks();

  const blob = new Blob(chunks, { type: recorderType.split(";")[0] || "audio/webm" });
  chunks = [];

  if (blob.size === 0) {
    setMessage("Recording was empty. Nothing was sent.");
    setIdleControls();
    return;
  }

  setState("Uploading…", false);
  await sendRecording(blob);
  setIdleControls();
}

function buildRequestDebug(file) {
  const headers = {
    "Content-Type": "multipart/form-data (boundary set by the browser)",
  };
  const payload = {
    fieldName: FILE_FIELD,
    filename: file.name,
    contentType: file.type || "audio/webm",
    sizeBytes: file.size,
  };

  return [
    `Method: POST`,
    `Path: ${TRANSCRIBE_PATH}`,
    "",
    "Headers:",
    JSON.stringify(headers, null, 2),
    "",
    "Payload:",
    JSON.stringify(payload, null, 2),
  ].join("\n");
}

function formatResponseBody(text) {
  if (!text) {
    return "(empty body)";
  }
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch (_error) {
    return text;
  }
}

function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function showResponseDebug(statusLine, bodyText) {
  responseDebug.textContent = [
    `Status: ${statusLine}`,
    "",
    formatResponseBody(bodyText),
  ].join("\n");
  responseBox.hidden = false;
}

function transcriptPath(filename) {
  return `${TRANSCRIBE_PATH}/${encodeURIComponent(filename)}`;
}

async function waitForTranscript(filename) {
  setState("Waiting for transcription…", false);
  const path = transcriptPath(filename);

  for (let attempt = 1; attempt <= POLL_MAX_ATTEMPTS; attempt += 1) {
    try {
      const response = await fetch(path);
      const text = await response.text();
      if (isLocalEnvironment()) {
        showResponseDebug(
          `${response.status} ${response.statusText} (GET ${path}, attempt ${attempt})`,
          text,
        );
      }

      if (response.ok) {
        const payload = JSON.parse(text);
        transcriptEl.textContent = payload.text || "";
        transcriptBox.hidden = false;
        return;
      }

      if (response.status === 404) {
        await sleep(POLL_INTERVAL_MS);
        continue;
      }

      setMessage("Could not retrieve the transcription.");
      return;
    } catch (_error) {
      await sleep(POLL_INTERVAL_MS);
    }
  }

  setMessage("The transcription is taking too long. Try again.");
}

async function sendRecording(blob) {
  const file = new File([blob], RECORDING_FILENAME, {
    type: blob.type || "audio/webm",
  });
  const formData = new FormData();
  formData.append(FILE_FIELD, file);

  if (isLocalEnvironment()) {
    requestDebug.textContent = buildRequestDebug(file);
    requestBox.hidden = false;
  } else {
    requestBox.hidden = true;
    requestDebug.textContent = "";
  }

  try {
    const response = await fetch(TRANSCRIBE_PATH, {
      method: "POST",
      body: formData,
    });
    const text = await response.text();
    showResponseDebug(`${response.status} ${response.statusText}`, text);

    if (response.status !== 202) {
      setMessage("The transcription request was not accepted.");
      return;
    }

    const payload = JSON.parse(text);
    if (!payload.filename) {
      setMessage("The transcription request did not return a filename.");
      return;
    }

    await waitForTranscript(payload.filename);
  } catch (_error) {
    showResponseDebug("(request failed)", "Could not reach the transcription API.");
    setMessage("Could not reach the transcription API.");
  }
}

startButton.addEventListener("click", () => {
  startRecording();
});
stopButton.addEventListener("click", () => {
  stopRecording();
});
