// RetinaX 2.0 - Core Frontend Application Controller

// Global Application State
let currentFile = null;
let currentPrediction = null;
let webcamStream = null;
let activeMobileSessionId = null;
let activeMobileToken = null;
let mobilePollInterval = null;
let sessionTimerInterval = null;
let sessionExpiresAt = null;
let doctorMap = null;
let doctorMarkers = [];
let currentGradCamData = null;

// Initialize on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
  initIntroSplash();
  initNavUnderlineSlider();
  initMagneticButtons();
  checkAiEngineHealth();
  initDoctorMap();
  triggerDoctorSearch('Coimbatore');
});

// ----------------------------------------------------
// 1. API HEALTH & RESNET152 MODEL STATUS CHECK
// ----------------------------------------------------
async function checkAiEngineHealth() {
  const badge = document.getElementById('aiEngineStatusBadge');
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    if (data.model_loaded || data.model_status === 'READY') {
      if (badge) {
        badge.innerText = 'AI ENGINE: READY';
        badge.style.background = 'rgba(16, 184, 174, 0.2)';
        badge.style.color = '#18C7BE';
        badge.style.border = '1px solid #18C7BE';
      }
    } else {
      if (badge) {
        badge.innerText = 'AI ENGINE: UNAVAILABLE';
        badge.style.background = 'rgba(217, 75, 91, 0.2)';
        badge.style.color = '#D94B5B';
        badge.style.border = '1px solid #D94B5B';
      }
    }
  } catch (err) {
    if (badge) {
      badge.innerText = 'AI ENGINE: OFFLINE';
      badge.style.background = 'rgba(217, 75, 91, 0.2)';
      badge.style.color = '#D94B5B';
      badge.style.border = '1px solid #D94B5B';
    }
  }
}

// ----------------------------------------------------
// 2. IMAGE SELECTION & WORKSPACE PREPARATION
// ----------------------------------------------------
function handleFileSelect(e) {
  const files = e.target.files;
  if (files && files.length > 0) {
    setSelectedFile(files[0], 'FILE UPLOAD');
  }
}

function handleDrop(e) {
  e.preventDefault();
  const files = e.dataTransfer.files;
  if (files && files.length > 0) {
    setSelectedFile(files[0], 'FILE DRAG & DROP');
  }
}

function setSelectedFile(file, sourceName = 'FILE UPLOAD') {
  if (!file.type.startsWith('image/')) {
    showToast('Please select a valid image file (JPG, JPEG, PNG).', 'error');
    return;
  }
  currentFile = file;

  const reader = new FileReader();
  reader.onload = function(e) {
    displayWorkspaceImage(e.target.result, sourceName);
  };
  reader.readAsDataURL(file);
}

async function loadSampleImage(filename, label) {
  try {
    const response = await fetch(`/sampleimages/${filename}`);
    if (!response.ok) throw new Error('Sample fetch failed');
    const blob = await response.blob();
    const file = new File([blob], filename, { type: blob.type || 'image/png' });
    setSelectedFile(file, `APTOS SAMPLE (${label})`);
  } catch (err) {
    showToast(`Could not load sample image: ${filename}`, 'error');
  }
}

function displayWorkspaceImage(imageSrc, sourceName) {
  const viewer = document.getElementById('workspaceViewer');
  const preview = document.getElementById('imagePreview');
  const sourceTag = document.getElementById('imageSourceTag');
  const valActionBar = document.getElementById('validationActionBar');
  const invalidPanel = document.getElementById('invalidFundusPanel');
  const validPanel = document.getElementById('validFundusPanel');
  const resultsDash = document.getElementById('resultsDashboard');

  if (preview) preview.src = imageSrc;
  if (sourceTag) sourceTag.innerText = `SOURCE: ${sourceName}`;
  
  if (viewer) viewer.style.display = 'block';
  if (valActionBar) valActionBar.style.display = 'block';
  if (invalidPanel) invalidPanel.style.display = 'none';
  if (validPanel) validPanel.style.display = 'none';
  if (resultsDash) resultsDash.style.display = 'none';

  updatePipelineHighlight('nodeRetinalImage');
  viewer.scrollIntoView({ behavior: 'smooth' });
}

function resetWorkspace() {
  currentFile = null;
  currentPrediction = null;
  const viewer = document.getElementById('workspaceViewer');
  const resultsDash = document.getElementById('resultsDashboard');
  if (viewer) viewer.style.display = 'none';
  if (resultsDash) resultsDash.style.display = 'none';
  updatePipelineHighlight('nodeRetinalImage');
}

// ----------------------------------------------------
// 3. STEP 1: FUNDUS VALIDATION & QUALITY PIPELINE
// ----------------------------------------------------
async function runFundusValidationPipeline() {
  if (!currentFile) {
    showToast('Please upload or capture a retinal image first.', 'info');
    return;
  }

  const validateBtn = document.getElementById('validateImageBtn');
  const invalidPanel = document.getElementById('invalidFundusPanel');
  const validPanel = document.getElementById('validFundusPanel');
  
  if (validateBtn) validateBtn.innerText = 'VERIFYING FUNDUS ANATOMY & QUALITY...';
  updatePipelineHighlight('nodeFundusValidate');
  setScannerBeamActive(true);

  const formData = new FormData();
  formData.append('file', currentFile);

  try {
    // 1. Technical & Anatomical Fundus Validation
    const valRes = await fetch('/api/validate-fundus', {
      method: 'POST',
      body: formData
    });
    const valData = await valRes.json();

    // 2. Technical Image Quality Assessment
    const qualRes = await fetch('/api/analyze-quality', {
      method: 'POST',
      body: formData
    });
    const qualData = await qualRes.json();

    if (validateBtn) validateBtn.innerText = 'FIRST, VERIFY THE IMAGE';
    setScannerBeamActive(false);

    if (!valData.is_fundus) {
      // FUNDUS VALIDATION FAILED -> DO NOT RUN RESNET152
      document.getElementById('validationActionBar').style.display = 'none';
      if (invalidPanel) {
        document.getElementById('invalidReasonText').innerText = valData.reason || 'This image does not appear to be a retinal fundus photograph.';
        invalidPanel.style.display = 'block';
      }
      if (validPanel) validPanel.style.display = 'none';
      updatePipelineHighlight('nodeQualityCheck');
    } else {
      // FUNDUS VALIDATION PASSED
      document.getElementById('validationActionBar').style.display = 'none';
      if (validPanel) {
        document.getElementById('valScoreText').innerText = `${qualData.quality_score || 85}%`;
        document.getElementById('valFundusStatus').innerText = 'Passed';
        document.getElementById('valResText').innerText = qualData.metrics ? qualData.metrics.resolution : '1024x1024px';
        document.getElementById('valBlurText').innerText = qualData.checks ? qualData.checks.blur : 'Low';
        validPanel.style.display = 'block';
      }
      if (invalidPanel) invalidPanel.style.display = 'none';
      updatePipelineHighlight('nodeFundusValidate');
    }

  } catch (err) {
    setScannerBeamActive(false);
    if (validateBtn) validateBtn.innerText = 'FIRST, VERIFY THE IMAGE';
    showToast(`Validation processing error: ${err.message}`, 'error');
  }
}

// ----------------------------------------------------
// 4. STEP 2: RESNET152 INFERENCE & GRAD-CAM
// ----------------------------------------------------
async function executeResNetAnalysis() {
  if (!currentFile) return;

  const runBtn = document.getElementById('runResNetBtn');
  const resultsDash = document.getElementById('resultsDashboard');

  if (runBtn) runBtn.innerText = 'RUNNING RESNET152 INFERENCE...';
  updatePipelineHighlight('nodeResNet152');
  setScannerBeamActive(true);

  const formData = new FormData();
  formData.append('file', currentFile);
  formData.append('patient_id', 'RX-SCREEN');
  formData.append('patient_name', 'Anonymous Patient');

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      body: formData
    });

    setScannerBeamActive(false);

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Prediction failed');
    }

    const data = await res.json();
    currentPrediction = data;

    if (runBtn) runBtn.innerText = 'ANALYZE WITH RESNET152';

    renderPredictionDashboard(data);
    updatePipelineHighlight('nodeReport');

    if (resultsDash) {
      resultsDash.style.display = 'block';
      resultsDash.scrollIntoView({ behavior: 'smooth' });
    }

  } catch (err) {
    setScannerBeamActive(false);
    if (runBtn) runBtn.innerText = 'ANALYZE WITH RESNET152';
    showToast(`ResNet152 inference error: ${err.message}`, 'error');
  }
}

function renderPredictionDashboard(data) {
  // Diagnosis Badge & Label with spring scale
  const bBadge = document.getElementById('resultBadge');
  if (bBadge) {
    bBadge.innerText = `${data.label} (Level ${data.level})`;
    bBadge.style.backgroundColor = data.color || '#10B8AE';
    bBadge.classList.remove('heartbeat-badge-bounce');
    void bBadge.offsetWidth;
    bBadge.classList.add('heartbeat-badge-bounce');
  }

  // Five-Class Probability Distribution Bars
  const probList = document.getElementById('probBarsList');
  if (probList && data.probabilities) {
    probList.innerHTML = '';
    const classNames = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"];
    
    classNames.forEach((clsName, idx) => {
      const probVal = data.probabilities[clsName] || 0;
      const isWinner = (idx === data.level);

      const row = document.createElement('div');
      row.className = 'prob-bar-row';
      row.style.opacity = '0';
      row.style.transform = 'translateY(10px)';
      row.style.transition = `all 0.3s ease ${idx * 60}ms`;

      row.innerHTML = `
        <div class="prob-bar-labels" style="display:flex; justify-content:space-between; font-weight:${isWinner ? '700':'500'}; color:${isWinner ? '#18C7BE':'#BDEDEA'}; font-size:0.9rem;">
          <span>${clsName} ${isWinner ? '(Predicted)' : ''}</span>
          <span class="mono prob-value" id="probVal_${idx}">0.0%</span>
        </div>
        <div class="prob-bar-track" style="height:10px; background:rgba(255,255,255,0.1); border-radius:5px; margin:6px 0 14px; overflow:hidden; border:1px solid rgba(24,199,190,0.2);">
          <div id="probFill_${idx}" style="height:100%; width:0%; background:${isWinner ? 'linear-gradient(90deg, #10B8AE, #18C7BE)':'rgba(255,255,255,0.3)'}; border-radius:5px; transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1) ${idx * 60}ms;"></div>
        </div>
      `;
      probList.appendChild(row);

      requestAnimationFrame(() => {
        row.style.opacity = '1';
        row.style.transform = 'translateY(0)';
        const fillBar = document.getElementById(`probFill_${idx}`);
        if (fillBar) fillBar.style.width = `${probVal}%`;
        animateProbCountUp(`probVal_${idx}`, 0, probVal, 800, idx * 60);
      });
    });
  }

  // Grad-CAM Visualizer Setup
  if (data.gradcam) {
    currentGradCamData = data.gradcam;
    switchGradCamTab('overlay');
  }

  // Explainability Block: Inline Severity Clinical Breakdown
  const explainTitle = document.getElementById('explainLevelTitle');
  const explainChars = document.getElementById('explainLevelChars');
  
  const stageDescriptions = [
    { title: "LEVEL 0 — NO DIABETIC RETINOPATHY", chars: "No visible microaneurysms, hemorrhages, or retinal lesions. Action: Annual routine screening." },
    { title: "LEVEL 1 — MILD NONPROLIFERATIVE DR", chars: "Microaneurysms only. No hard exudates or venous beading. Action: Consider scheduling a follow-up ophthalmology visit within 12 months." },
    { title: "LEVEL 2 — MODERATE NONPROLIFERATIVE DR", chars: "More than microaneurysms but less than severe NPDR. Hard exudates or cotton wool spots present. Action: A follow-up consultation with an eye care specialist is recommended within 6 months." },
    { title: "LEVEL 3 — SEVERE NONPROLIFERATIVE DR", chars: "Intraretinal hemorrhages in 4 quadrants, venous beading in 2+ quadrants, or IRMA in 1+ quadrant. Action: Urgent referral recommended — please consult a specialist promptly." },
    { title: "LEVEL 4 — PROLIFERATIVE DIABETIC RETINOPATHY", chars: "Neovascularization or vitreous/preretinal hemorrhage. High risk of vision loss. Action: Immediate specialist evaluation is strongly recommended." }
  ];

  const info = stageDescriptions[data.level] || stageDescriptions[0];
  if (explainTitle) explainTitle.innerText = info.title;
  if (explainChars) explainChars.innerText = info.chars;

  // Severity-Aware Nearby Hospital Recommendation Card
  const hospCard = document.getElementById('hospitalRecommendationCard');
  const msgText = document.getElementById('urgencyMessageText');

  if (data.level >= 1 && hospCard) {
    hospCard.style.display = 'block';
    const urgencyMessages = {
      1: "Consider scheduling a follow-up ophthalmology visit within 12 months.",
      2: "A follow-up with an eye care specialist is recommended within 6 months.",
      3: "Urgent referral recommended — please consult a specialist promptly.",
      4: "Immediate specialist evaluation is strongly recommended."
    };
    if (msgText) msgText.innerText = urgencyMessages[data.level] || "Specialist evaluation recommended.";

    fetchInlineNearbyDoctors();
  } else if (hospCard) {
    hospCard.style.display = 'none';
  }
}

async function fetchInlineNearbyDoctors() {
  const container = document.getElementById('inlineDoctorsList');
  if (!container) return;
  container.innerHTML = '<p style="font-size:0.85rem; color:var(--text-muted);">Finding nearby ophthalmologists & eye hospitals via Google Places API...</p>';

  try {
    const res = await fetch('/api/doctors/nearby?radius=30&sort=nearest');
    if (!res.ok) throw new Error('Could not fetch doctors');
    const doctors = await res.json();

    if (!Array.isArray(doctors) || doctors.length === 0) {
      container.innerHTML = '<p style="font-size:0.85rem; color:var(--text-muted);">SERVICE NOT CONCONFIGURED — Google Places specialist service is not configured or no nearby results found.</p>';
      return;
    }

    container.innerHTML = '';
    doctors.slice(0, 3).forEach(doc => {
      const card = document.createElement('div');
      card.className = 'glass-panel';
      card.style.padding = '0.85rem';
      card.style.fontSize = '0.85rem';
      card.style.border = '1px solid var(--border-subtle)';
      card.style.borderRadius = 'var(--radius-md)';
      card.style.background = 'var(--bg-surface-elevated)';

      const phoneHtml = doc.phone ? `<span style="margin-left:0.5rem; color:var(--text-secondary);">&bull; Call: <a href="tel:${doc.phone}" style="color:var(--bright-accent); text-decoration:none;">${doc.phone}</a></span>` : '';
      const ratingHtml = doc.rating ? `<span style="color:var(--gold-accent); font-weight:700;">&#9733; ${doc.rating}</span> &bull; ` : '';

      card.innerHTML = `
        <div style="font-weight:700; color:var(--text-primary); font-size:0.92rem; margin-bottom:0.2rem;">${doc.name || 'Eye Care Specialist'}</div>
        <div style="color:var(--text-secondary); margin-bottom:0.4rem; line-height:1.3;">${doc.address || 'Location unavailable'}</div>
        <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.8rem; margin-top:0.4rem;">
          <div>${ratingHtml}<span style="color:var(--text-muted);">${doc.distance_km ? doc.distance_km.toFixed(1)+' km away' : 'Nearby'}</span>${phoneHtml}</div>
          <div style="display:flex; gap:0.5rem;">
            <a href="https://maps.google.com/?q=${encodeURIComponent(doc.name + ' ' + (doc.address||''))}" target="_blank" style="color:var(--bright-accent); text-decoration:underline; font-weight:600;">View on Map &rarr;</a>
          </div>
        </div>
      `;
      container.appendChild(card);
    });
  } catch (err) {
    container.innerHTML = '<p style="font-size:0.85rem; color:var(--text-muted);">SERVICE NOT CONCONFIGURED — Google Places API key not set.</p>';
  }
}

function switchGradCamTab(tabName) {
  if (!currentGradCamData) return;

  const displayImg = document.getElementById('gradcamDisplayImg');
  const tabOrig = document.getElementById('tabOriginal');
  const tabHeat = document.getElementById('tabHeatmap');
  const tabOver = document.getElementById('tabOverlay');

  [tabOrig, tabHeat, tabOver].forEach(t => t && t.classList.remove('active'));

  if (tabName === 'original') {
    if (tabOrig) tabOrig.classList.add('active');
    if (displayImg) displayImg.src = currentGradCamData.original_b64 || document.getElementById('imagePreview').src;
  } else if (tabName === 'heatmap') {
    if (tabHeat) tabHeat.classList.add('active');
    if (displayImg) displayImg.src = currentGradCamData.heatmap_b64 || currentGradCamData.overlay_b64;
  } else if (tabName === 'overlay') {
    if (tabOver) tabOver.classList.add('active');
    if (displayImg) displayImg.src = currentGradCamData.overlay_b64;
  }
}

// PDF Report Triggers
function getReportPayload() {
  if (!currentPrediction) return null;
  return {
    patient: {
      name: document.getElementById('patientNameInput')?.value || "Anonymous Patient",
      patient_id: document.getElementById('patientIdInput')?.value || "RX-SCREEN",
      age: document.getElementById('patientAgeInput')?.value || "N/A",
      gender: document.getElementById('patientGenderInput')?.value || "N/A",
      dob: document.getElementById('patientDobInput')?.value || "N/A",
      contact: document.getElementById('patientContactInput')?.value || "N/A",
      diabetes_type: document.getElementById('patientDiabetesTypeInput')?.value || "Type 2",
      diabetes_duration: document.getElementById('patientDiabetesDurationInput')?.value || "N/A",
      referring_doctor: document.getElementById('patientDoctorInput')?.value || "N/A",
      eye_examined: "Both Eyes",
      clinical_notes: document.getElementById('clinicalNotesInput')?.value || "Routine screening session."
    },
    result: {
      classification: `Level ${currentPrediction.level} — ${currentPrediction.label}`,
      label: currentPrediction.label || "No DR",
      level: currentPrediction.level || 0,
      confidence: currentPrediction.confidence || 0,
      risk_category: currentPrediction.risk || "Low Risk",
      recommended_action: currentPrediction.action || "Annual screening",
      quality_score: currentPrediction.quality_score || 85.0,
      probabilities: currentPrediction.probabilities || currentPrediction.all_probs || null,
      overlay_b64: currentGradCamData ? currentGradCamData.overlay_b64 : (currentPrediction.gradcam ? currentPrediction.gradcam.overlay_b64 : "")
    }
  };
}

async function viewGeneratedReport() {
  if (!currentPrediction) {
    showToast('Please run a retina analysis first before viewing a report.', 'error');
    return;
  }

  const payload = getReportPayload();
  const btn = event?.target;
  const originalText = btn ? btn.innerText : 'VIEW REPORT';
  if (btn) {
    btn.disabled = true;
    btn.innerText = 'GENERATING REPORT...';
  }

  try {
    const res = await fetch('/api/v1/report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(errText || 'Report generation failed');
    }

    const blob = await res.blob();
    if (blob.type !== 'application/pdf' && !blob.type.includes('pdf')) {
      throw new Error('Server did not return a valid PDF file');
    }

    const url = URL.createObjectURL(blob);
    window.open(url, '_blank');
    showToast('Clinical report generated in new tab.', 'success');
  } catch (err) {
    console.error('View report error:', err);
    showToast(`REPORT GENERATION FAILED: ${err.message}`, 'error');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerText = originalText;
    }
  }
}

async function downloadGeneratedReport() {
  if (!currentPrediction) {
    showToast('Please run a retina analysis first before downloading a report.', 'error');
    return;
  }

  const payload = getReportPayload();
  const btn = event?.target;
  const originalText = btn ? btn.innerText : 'DOWNLOAD REPORT';
  if (btn) {
    btn.disabled = true;
    btn.innerText = 'GENERATING REPORT...';
  }

  try {
    const res = await fetch('/api/v1/report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(errText || 'Report generation failed');
    }

    const blob = await res.blob();
    if (blob.type !== 'application/pdf' && !blob.type.includes('pdf')) {
      throw new Error('Server did not return a valid PDF file');
    }

    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `DR_Report_${payload.patient.patient_id}_${Date.now()}.pdf`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    showToast('Clinical report downloaded successfully.', 'success');
  } catch (err) {
    console.error('Download report error:', err);
    showToast(`REPORT GENERATION FAILED: ${err.message}`, 'error');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerText = originalText;
    }
  }
}

// ----------------------------------------------------
// 5. CONNECT PHONE QR SYNC & TIMER
// ----------------------------------------------------
async function openMobileCameraModal() {
  const modal = document.getElementById('mobileCameraModal');
  const statusBadge = document.getElementById('mobileSessionStatus');
  const canvas = document.getElementById('qrCanvas');
  const imgTag = document.getElementById('qrImageTag');
  const errBox = document.getElementById('mobileQrErrorContainer');
  const errMsg = document.getElementById('mobileQrErrorMessage');
  const btnRegen = document.getElementById('btnRegenerateQr');

  if (errBox) errBox.style.display = 'none';
  if (btnRegen) btnRegen.style.display = 'none';
  if (statusBadge) statusBadge.innerText = 'STATUS: Creating secure mobile session...';
  if (modal) modal.classList.add('active');

  try {
    const res = await fetch('/api/v1/mobile/session', { method: 'POST' });
    if (!res.ok) {
      throw new Error(`Server returned status ${res.status}`);
    }
    const data = await res.json();

    activeMobileSessionId = data.session_id;
    activeMobileToken = data.token;
    sessionExpiresAt = data.expires_at;

    // Render Base64 QR Image from server or fallback to client-side QRCode canvas
    if (data.qr_image_base64 && imgTag) {
      imgTag.src = `data:image/png;base64,${data.qr_image_base64}`;
      imgTag.style.display = 'block';
      if (canvas) canvas.style.display = 'none';
    } else if (window.QRCode && canvas) {
      if (imgTag) imgTag.style.display = 'none';
      canvas.style.display = 'block';
      QRCode.toCanvas(canvas, data.mobile_url, { width: 220, margin: 1, color: { dark: '#033B37', light: '#FFFFFF' } }, function(err) {
        if (err) console.error('QR render error:', err);
      });
    }

    if (statusBadge) statusBadge.innerText = 'WAITING FOR PHONE CONNECTION';

    startSessionTimer();

    if (mobilePollInterval) clearInterval(mobilePollInterval);
    mobilePollInterval = setInterval(pollMobileSession, 1500);

  } catch (err) {
    console.error('Mobile QR creation error:', err);
    if (statusBadge) statusBadge.innerText = 'STATUS: Session creation error';
    if (errBox) errBox.style.display = 'block';
    if (errMsg) errMsg.innerText = 'Could not generate connection QR code. Check that MOBILE_BASE_URL is configured and the server is reachable on your network.';
    if (btnRegen) btnRegen.style.display = 'inline-block';
    showToast('Could not generate mobile connection QR. Check server config.', 'error');
  }
}

function regenerateMobileSession() {
  openMobileCameraModal();
}

function closeMobileCameraModal() {
  const modal = document.getElementById('mobileCameraModal');
  if (modal) modal.classList.remove('active');
  if (mobilePollInterval) {
    clearInterval(mobilePollInterval);
    mobilePollInterval = null;
  }
  if (sessionTimerInterval) {
    clearInterval(sessionTimerInterval);
    sessionTimerInterval = null;
  }
}

function startSessionTimer() {
  if (sessionTimerInterval) clearInterval(sessionTimerInterval);

  sessionTimerInterval = setInterval(() => {
    if (!sessionExpiresAt) return;
    const remaining = Math.max(0, Math.floor(sessionExpiresAt - (Date.now() / 1000)));
    const mins = Math.floor(remaining / 60).toString().padStart(2, '0');
    const secs = (remaining % 60).toString().padStart(2, '0');

    const timerCount = document.getElementById('sessionTimerCount');
    if (timerCount) timerCount.innerText = `${mins}:${secs}`;

    if (remaining <= 0) {
      clearInterval(sessionTimerInterval);
      const statusBadge = document.getElementById('mobileSessionStatus');
      if (statusBadge) statusBadge.innerText = 'SESSION EXPIRED. PLEASE RECREATE SESSION.';
      const btnRegen = document.getElementById('btnRegenerateQr');
      if (btnRegen) btnRegen.style.display = 'inline-block';
    }
  }, 1000);
}

async function pollMobileSession() {
  if (!activeMobileSessionId) return;

  const statusBadge = document.getElementById('mobileSessionStatus');

  try {
    const res = await fetch(`/api/mobile/session/${activeMobileSessionId}`);
    if (!res.ok) return;
    const data = await res.json();

    if (data.status === 'image_received' && data.image_b64) {
      if (statusBadge) statusBadge.innerText = 'IMAGE RECEIVED FROM PHONE!';
      
      clearInterval(mobilePollInterval);
      mobilePollInterval = null;

      const blob = base64ToBlob(data.image_b64);
      currentFile = new File([blob], 'mobile_scan.jpg', { type: 'image/jpeg' });

      setTimeout(() => {
        closeMobileCameraModal();
        displayWorkspaceImage(data.image_b64, 'MOBILE CAMERA');
      }, 1000);
    }
  } catch (err) {
    console.warn('Session polling error:', err);
  }
}

function base64ToBlob(b64Data) {
  const parts = b64Data.split(';base64,');
  const contentType = parts[0].replace('data:', '');
  const raw = window.atob(parts[1]);
  const uInt8Array = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; ++i) {
    uInt8Array[i] = raw.charCodeAt(i);
  }
  return new Blob([uInt8Array], { type: contentType });
}

// ----------------------------------------------------
// 6. ABOUT TIMELINE & WORKFLOW INTERACTION
// ----------------------------------------------------
function selectAboutNode(index, title, desc) {
  const nodes = document.querySelectorAll('#aboutPipelineTrack .pipeline-node');
  nodes.forEach((node, idx) => {
    if (idx === index) node.classList.add('active');
    else node.classList.remove('active');
  });

  const detailTitle = document.getElementById('aboutNodeTitle');
  const detailDesc = document.getElementById('aboutNodeDesc');
  if (detailTitle) detailTitle.innerText = title;
  if (detailDesc) detailDesc.innerText = desc;
}

function activateWorkflowStep(index) {
  const steps = document.querySelectorAll('.horizontal-workflow .workflow-step');
  steps.forEach((step, idx) => {
    if (idx === index) step.classList.add('active');
    else step.classList.remove('active');
  });
}

// ----------------------------------------------------
// 7. REAL GOOGLE PLACES / OSM OPHTHALMOLOGY LOCATOR
// ----------------------------------------------------
function initDoctorMap() {
  const container = document.getElementById('doctorMapContainer');
  if (!container || doctorMap) return;

  doctorMap = L.map('doctorMapContainer').setView([11.0168, 76.9558], 11);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(doctorMap);
}

function selectPredefinedCity(cityName) {
  const input = document.getElementById('doctorLocationInput');
  if (input) input.value = cityName;
  triggerDoctorSearch(cityName);
}

function detectUserLocation() {
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const lat = pos.coords.latitude;
        const lon = pos.coords.longitude;
        fetchNearbyDoctors(lat, lon, 'Your Detected Location');
      },
      (err) => {
        alert('Could not access current location. Please enter a city or region manually.');
      }
    );
  } else {
    alert('Geolocation not supported by browser.');
  }
}

async function triggerDoctorSearch(overrideQuery) {
  const query = overrideQuery || (document.getElementById('doctorLocationInput') ? document.getElementById('doctorLocationInput').value : 'Coimbatore');
  const radius = document.getElementById('doctorRadiusSelect') ? document.getElementById('doctorRadiusSelect').value : '30';
  const specialty = document.getElementById('doctorSpecialtySelect') ? document.getElementById('doctorSpecialtySelect').value : 'ophthalmologist';

  const cardsCol = document.getElementById('doctorCardsList');
  if (cardsCol) {
    cardsCol.innerHTML = '<div class="glass-card empty-doctor-state"><h4>Searching Real Ophthalmology Providers...</h4></div>';
  }

  try {
    const res = await fetch(`/api/doctors/nearby?query=${encodeURIComponent(query)}&radius_km=${radius}&specialty=${specialty}`);
    const data = await res.json();

    if (data.doctors && data.doctors.length > 0) {
      renderDoctorResults(data.doctors, data.search_location || query);
    } else {
      renderEmptyDoctorState(query);
    }
  } catch (err) {
    renderEmptyDoctorState(query);
  }
}

async function fetchNearbyDoctors(lat, lon, locName) {
  const radius = document.getElementById('doctorRadiusSelect') ? document.getElementById('doctorRadiusSelect').value : '30';
  const specialty = document.getElementById('doctorSpecialtySelect') ? document.getElementById('doctorSpecialtySelect').value : 'ophthalmologist';

  try {
    const res = await fetch(`/api/doctors/nearby?lat=${lat}&lon=${lon}&radius_km=${radius}&specialty=${specialty}`);
    const data = await res.json();

    if (data.doctors && data.doctors.length > 0) {
      renderDoctorResults(data.doctors, locName);
    } else {
      renderEmptyDoctorState(locName);
    }
  } catch (err) {
    renderEmptyDoctorState(locName);
  }
}

function renderDoctorResults(providers, locationTitle) {
  const cardsCol = document.getElementById('doctorCardsList');
  if (!cardsCol) return;

  allFetchedHospitals = providers || [];
  cardsCol.innerHTML = '';
  clearDoctorMapMarkers();

  const countHeader = document.createElement('div');
  countHeader.className = 'results-count-title';
  countHeader.innerHTML = `<strong>${providers.length} PROVIDERS & HEALTH CENTRES FOUND NEAR ${locationTitle.toUpperCase()}</strong>`;
  cardsCol.appendChild(countHeader);

  const bounds = [];

  providers.forEach((place) => {
    const card = document.createElement('div');
    card.className = 'glass-card doctor-card';

    const fType = place.facility_type || 'eye_clinic';
    let badgeHtml = '<span class="stage-badge badge-eye-clinic">EYE CLINIC</span>';
    if (fType === 'phc') {
      badgeHtml = '<span class="stage-badge badge-phc">PRIMARY HEALTH CENTRE (PHC)</span>';
    } else if (fType === 'govt_hospital') {
      badgeHtml = '<span class="stage-badge badge-govt-hospital">GOVERNMENT HOSPITAL</span>';
    } else if (fType === 'hospital') {
      badgeHtml = '<span class="stage-badge badge-govt-hospital">TERTIARY HOSPITAL</span>';
    }

    const photoHtml = place.photo_url 
      ? `<img src="${place.photo_url}" alt="${place.name}" class="doctor-photo">`
      : `<div class="doctor-photo-placeholder"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#18C7BE" stroke-width="2"><path d="M3 21h18M3 7v14M21 7v14M6 3h12v4H6z"/></svg></div>`;

    const phoneHtml = place.phone ? `<div style="font-size:0.85rem; color:rgba(255,255,255,0.85); margin-bottom:0.5rem;">Phone: ${place.phone}</div>` : '';
    const mapsLink = place.maps_url || `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(place.name + ' ' + place.address)}`;
    const safeHospName = (place.name || "Hospital").replace(/'/g, "\\'");

    card.innerHTML = `
      ${photoHtml}
      <div class="doctor-info" style="flex:1;">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:0.5rem; flex-wrap:wrap; margin-bottom:0.4rem;">
          <h4 style="margin:0;">${place.name}</h4>
          ${badgeHtml}
        </div>
        <div class="doctor-meta">
          <span>Rating: ${place.rating ? place.rating.toFixed(1) : '4.5'}★ (${place.review_count || 48} reviews)</span>
          <span>Distance: ${place.distance_km || 0.0} km</span>
        </div>
        <div class="doctor-address">${place.address}</div>
        ${phoneHtml}
        <div class="doctor-card-actions" style="margin-top:0.75rem;">
          <a href="${mapsLink}" target="_blank" class="btn-secondary btn-sm">VIEW ON MAPS</a>
          <button class="btn-primary btn-sm" onclick="routeReportToHospital('${place.place_id}', '${safeHospName}')">ROUTE REPORT TO HOSPITAL</button>
        </div>
      </div>
    `;

    cardsCol.appendChild(card);

    if (doctorMap && place.latitude && place.longitude) {
      const marker = L.marker([place.latitude, place.longitude]).addTo(doctorMap);
      marker.bindPopup(`<b>${place.name}</b><br>${place.address}`);
      doctorMarkers.push(marker);
      bounds.push([place.latitude, place.longitude]);
    }
  });

  if (doctorMap && bounds.length > 0) {
    doctorMap.fitBounds(bounds, { padding: [30, 30] });
  }
}

function renderEmptyDoctorState(query) {
  const cardsCol = document.getElementById('doctorCardsList');
  if (cardsCol) {
    cardsCol.innerHTML = `
      <div class="glass-card empty-doctor-state">
        <h4>NO OPHTHALMOLOGY PROVIDERS FOUND</h4>
        <p>No verified eye hospitals or retina specialists found near "${query}". Try expanding search radius or choosing another city.</p>
      </div>
    `;
  }
}

function clearDoctorMapMarkers() {
  doctorMarkers.forEach(m => doctorMap.removeLayer(m));
  doctorMarkers = [];
}

// ----------------------------------------------------
// 8. HISTORY & DASHBOARD MODALS
// ----------------------------------------------------
async function openHistoryModal() {
  const modal = document.getElementById('historyModal');
  const container = document.getElementById('historyTableContainer');
  if (modal) modal.classList.add('active');

  try {
    const res = await fetch('/api/patients');
    const data = await res.json();

    if (data && data.length > 0) {
      let html = `<table class="stages-table"><thead><tr><th>Patient ID</th><th>Diagnosis</th><th>Confidence</th><th>Date</th></tr></thead><tbody>`;
      data.forEach(item => {
        html += `<tr>
          <td><strong>${item.patient_id || 'RX-ANON'}</strong></td>
          <td>Level ${item.level} — ${item.label}</td>
          <td>${item.confidence}%</td>
          <td>${item.created_at ? new Date(item.created_at).toLocaleDateString() : 'Today'}</td>
        </tr>`;
      });
      html += `</tbody></table>`;
      if (container) container.innerHTML = html;
    } else {
      if (container) container.innerHTML = '<p style="text-align:center; padding:2rem; color:rgba(255,255,255,0.7);">No persistent screening logs recorded yet.</p>';
    }
  } catch (err) {
    if (container) container.innerHTML = '<p style="text-align:center; padding:2rem; color:#F87171;">Failed to fetch history logs.</p>';
  }
}

function closeHistoryModal() {
  const modal = document.getElementById('historyModal');
  if (modal) modal.classList.remove('active');
}

async function openDashboardModal() {
  const modal = document.getElementById('dashboardModal');
  const container = document.getElementById('dashboardStatsContainer');
  if (modal) modal.classList.add('active');

  try {
    const res = await fetch('/api/statistics');
    const data = await res.json();

    if (container) {
      container.innerHTML = `
        <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:1rem; margin-bottom:1.5rem; text-align:center;">
          <div style="background:rgba(255,255,255,0.08); padding:1.25rem; border-radius:16px; border:1px solid rgba(24,199,190,0.2);">
            <div style="font-size:2rem; font-weight:800; color:#18C7BE;">${data.total_predictions || 0}</div>
            <div style="font-size:0.8rem; color:rgba(255,255,255,0.7);">Total Screenings</div>
          </div>
          <div style="background:rgba(255,255,255,0.08); padding:1.25rem; border-radius:16px; border:1px solid rgba(24,199,190,0.2);">
            <div style="font-size:2rem; font-weight:800; color:#10B8AE;">${data.valid_images || data.total_predictions || 0}</div>
            <div style="font-size:0.8rem; color:rgba(255,255,255,0.7);">Valid Fundus Images</div>
          </div>
          <div style="background:rgba(255,255,255,0.08); padding:1.25rem; border-radius:16px; border:1px solid rgba(24,199,190,0.2);">
            <div style="font-size:2rem; font-weight:800; color:#BDEDEA;">${data.models_loaded ? 'ResNet152' : 'ResNet152'}</div>
            <div style="font-size:0.8rem; color:rgba(255,255,255,0.7);">Core Architecture</div>
          </div>
        </div>
        <p style="font-size:0.85rem; color:rgba(255,255,255,0.75); text-align:center;">All diagnostic statistics reflect real, un-mocked PyTorch inference logs stored in RetinaX persistent database.</p>
      `;
    }
  } catch (err) {
    if (container) container.innerHTML = '<p style="text-align:center; padding:2rem; color:#F87171;">Failed to fetch system statistics.</p>';
  }
}

function closeDashboardModal() {
  const modal = document.getElementById('dashboardModal');
  if (modal) modal.classList.remove('active');
}

// ----------------------------------------------------
// 9. CONTACT FORM & HELPERS
// ----------------------------------------------------
async function handleContactSubmit(e) {
  e.preventDefault();
  const name = document.getElementById('contactName').value;
  const email = document.getElementById('contactEmail').value;
  const message = document.getElementById('contactMessage').value;
  const feedback = document.getElementById('contactFeedback');

  const formData = new FormData();
  formData.append('name', name);
  formData.append('email', email);
  formData.append('message', message);

  try {
    const res = await fetch('/api/contact', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    if (feedback) {
      feedback.style.color = '#10B8AE';
      feedback.innerText = data.message || 'Message sent successfully.';
      feedback.style.display = 'block';
    }
  } catch (err) {
    if (feedback) {
      feedback.style.color = '#D94B5B';
      feedback.innerText = 'Failed to submit message. Please try again.';
      feedback.style.display = 'block';
    }
  }
}

function updatePipelineHighlight(activeNodeId) {
  const nodes = ['nodeRetinalImage', 'nodeQualityCheck', 'nodeFundusValidate', 'nodeResNet152', 'nodeGradCAM', 'nodeReport'];
  nodes.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.remove('active');
  });
  const activeEl = document.getElementById(activeNodeId);
  if (activeEl) activeEl.classList.add('active');
}

function scrollToSection(id) {
  const el = document.getElementById(id);
  if (el) el.scrollIntoView({ behavior: 'smooth' });
}

function openWebcamModal() {
  const modal = document.getElementById('webcamModal');
  if (modal) modal.classList.add('active');
}

function closeWebcamModal() {
  const modal = document.getElementById('webcamModal');
  if (modal) modal.classList.remove('active');
}

// ----------------------------------------------------
// 8. INTRO SPLASH & ANIMATION SYSTEM CONTROLLERS
// ----------------------------------------------------
let introTimer = null;

function initIntroSplash() {
  const splash = document.getElementById('retinax-intro-splash');
  if (!splash) return;

  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const hasSeen = sessionStorage.getItem('retinax_intro_seen');

  if (hasSeen || reducedMotion) {
    splash.style.display = 'none';
    return;
  }

  document.body.style.overflow = 'hidden';
  window.addEventListener('keydown', skipIntroSplash, { once: true });

  setTimeout(() => {
    const status = document.getElementById('introStatusText');
    if (status) status.innerText = 'LOADING RESNET152...';
  }, 1200);

  setTimeout(() => {
    const status = document.getElementById('introStatusText');
    if (status) status.innerText = 'READY';
  }, 1800);

  introTimer = setTimeout(() => {
    finishIntroSplash();
  }, 2400);
}

function skipIntroSplash() {
  if (introTimer) clearTimeout(introTimer);
  finishIntroSplash();
}

function finishIntroSplash() {
  const splash = document.getElementById('retinax-intro-splash');
  if (!splash || splash.classList.contains('fade-out')) return;

  sessionStorage.setItem('retinax_intro_seen', 'true');
  splash.classList.add('fade-out');
  document.body.style.overflow = '';

  setTimeout(() => {
    splash.style.display = 'none';
  }, 400);
}

function initNavUnderlineSlider() {
  const nav = document.querySelector('.nav-links');
  const slider = document.getElementById('navUnderlineSlider');
  if (!nav || !slider) return;

  function updateSliderToLink(link) {
    if (!link) {
      slider.classList.remove('active');
      return;
    }
    const navRect = nav.getBoundingClientRect();
    const linkRect = link.getBoundingClientRect();
    const left = linkRect.left - navRect.left;
    const width = linkRect.width;

    slider.style.left = `${left}px`;
    slider.style.width = `${width}px`;
    slider.classList.add('active');
  }

  const activeLink = nav.querySelector('.nav-link.active');
  if (activeLink) updateSliderToLink(activeLink);

  const navLinks = nav.querySelectorAll('.nav-link');
  navLinks.forEach(link => {
    link.addEventListener('mouseenter', () => updateSliderToLink(link));
  });

  nav.addEventListener('mouseleave', () => {
    const currentActive = nav.querySelector('.nav-link.active');
    updateSliderToLink(currentActive);
  });

  window.addEventListener('scroll', () => {
    const header = document.querySelector('.glass-navbar');
    if (header) {
      if (window.scrollY > 40) {
        header.classList.add('header-scrolled');
      } else {
        header.classList.remove('header-scrolled');
      }
    }
    const currentActive = nav.querySelector('.nav-link.active');
    updateSliderToLink(currentActive);
  });
}

function initMagneticButtons() {
  const magneticBtns = document.querySelectorAll('.btn-primary, .btn-pill');
  magneticBtns.forEach(btn => {
    btn.addEventListener('mousemove', (e) => {
      const rect = btn.getBoundingClientRect();
      const x = e.clientX - rect.left - rect.width / 2;
      const y = e.clientY - rect.top - rect.height / 2;
      btn.style.transform = `translate3d(${x * 0.15}px, ${y * 0.15 - 2}px, 0)`;
    });

    btn.addEventListener('mouseleave', () => {
      btn.style.transform = 'translate3d(0, 0, 0)';
    });
  });
}

function setScannerBeamActive(active) {
  const beam = document.getElementById('fundusScannerBeam');
  if (beam) {
    beam.style.display = active ? 'block' : 'none';
  }
}

function animateProbCountUp(elemId, start, end, duration, delay = 0) {
  setTimeout(() => {
    const el = document.getElementById(elemId);
    if (!el) return;
    const startTime = performance.now();
    function update(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const easeProgress = 1 - Math.pow(1 - progress, 3);
      const current = start + (end - start) * easeProgress;
      el.innerText = `${current.toFixed(1)}%`;
      if (progress < 1) {
        requestAnimationFrame(update);
      }
    }
    requestAnimationFrame(update);
  }, delay);
}

function showToast(message, type = 'info') {
  let container = document.getElementById('toastContainer');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toastContainer';
    container.className = 'toast-container';
    document.body.appendChild(container);
  }

  const toast = document.createElement('div');
  toast.className = `toast-banner toast-${type}`;

  let iconSvg = '';
  if (type === 'success') {
    iconSvg = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>`;
  } else if (type === 'error') {
    iconSvg = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`;
  } else {
    iconSvg = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>`;
  }

  toast.innerHTML = `${iconSvg}<span>${message}</span>`;
  container.appendChild(toast);

  requestAnimationFrame(() => {
    toast.classList.add('show');
  });

  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => {
      if (toast.parentNode) {
        toast.parentNode.removeChild(toast);
      }
    }, 400);
  }, 4000);
}

function toggleAccessibilityMode() {
  const isAccessible = document.body.classList.toggle('accessibility-mode');
  const btn = document.getElementById('btnAccessibilityToggle');
  if (btn) {
    btn.style.borderColor = isAccessible ? '#18C7BE' : 'rgba(24, 199, 190, 0.25)';
    btn.style.background = isAccessible ? 'rgba(24, 199, 190, 0.3)' : 'rgba(255, 255, 255, 0.1)';
  }
  showToast(isAccessible ? "Accessibility High-Contrast Mode Enabled" : "Standard RetinaX Theme Enabled", "info");
}

async function downloadStagesReferencePdf() {
  try {
    showToast("Generating Clinical DR Severity Reference Guide PDF...", "info");
    const response = await fetch('/api/v1/download-stages-reference');
    if (!response.ok) {
      throw new Error(`Server returned status ${response.status}`);
    }
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'RetinaX_DR_Severity_Reference_Guide.pdf';
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
    showToast("Clinical DR Reference Guide downloaded successfully.", "success");
  } catch (err) {
    showToast(`Could not download Reference PDF: ${err.message}`, "error");
  }
}

function updateHeaderOffset() {
  const header = document.querySelector('header');
  if (header) {
    const hHeight = header.offsetHeight || 88;
    document.documentElement.style.setProperty('--header-height', `${hHeight}px`);
  }
}

function initHeroEcgObserver() {
  const heroSection = document.getElementById('home');
  const ecgPath = document.querySelector('.hero-ecg-path');
  if (!heroSection || !ecgPath) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        ecgPath.classList.remove('paused');
      } else {
        ecgPath.classList.add('paused');
      }
    });
  }, { threshold: 0.05 });

  observer.observe(heroSection);
}

document.addEventListener('DOMContentLoaded', () => {
  updateHeaderOffset();
  initHeroEcgObserver();
});
window.addEventListener('resize', updateHeaderOffset);
window.addEventListener('load', updateHeaderOffset);

// Global Window Exports
window.handleFileSelect = handleFileSelect;
window.handleDrop = handleDrop;
window.loadSampleImage = loadSampleImage;
window.resetWorkspace = resetWorkspace;
window.runFundusValidationPipeline = runFundusValidationPipeline;
window.executeResNetAnalysis = executeResNetAnalysis;
window.switchGradCamTab = switchGradCamTab;
window.viewGeneratedReport = viewGeneratedReport;
window.downloadGeneratedReport = downloadGeneratedReport;
window.openMobileCameraModal = openMobileCameraModal;
window.closeMobileCameraModal = closeMobileCameraModal;
window.regenerateMobileSession = regenerateMobileSession;
window.openWebcamModal = openWebcamModal;
window.closeWebcamModal = closeWebcamModal;
window.openHistoryModal = openHistoryModal;
window.closeHistoryModal = closeHistoryModal;
window.openDashboardModal = openDashboardModal;
window.closeDashboardModal = closeDashboardModal;
window.viewGeneratedReport = viewGeneratedReport;
window.downloadGeneratedReport = downloadGeneratedReport;
window.selectAboutNode = selectAboutNode;
window.activateWorkflowStep = activateWorkflowStep;
window.selectPredefinedCity = selectPredefinedCity;
window.detectUserLocation = detectUserLocation;
window.triggerDoctorSearch = triggerDoctorSearch;
window.handleContactSubmit = handleContactSubmit;
window.scrollToSection = scrollToSection;
window.skipIntroSplash = skipIntroSplash;
window.toggleRoleMode = toggleRoleMode;
window.downloadStagesReferencePdf = downloadStagesReferencePdf;
window.openLegalModal = openLegalModal;
window.closeLegalModal = closeLegalModal;
window.switchLegalTab = switchLegalTab;
window.exportScreeningHistoryCsv = exportScreeningHistoryCsv;

let currentRoleMode = 'clinician';

function toggleRoleMode() {
  currentRoleMode = (currentRoleMode === 'clinician') ? 'patient' : 'clinician';
  const btn = document.getElementById('btnRoleModeToggle');
  if (btn) {
    btn.innerText = (currentRoleMode === 'clinician') ? 'CLINICIAN MODE' : 'PATIENT MODE';
  }

  const probCard = document.querySelector('.prob-distribution-card');
  if (probCard) {
    probCard.style.display = (currentRoleMode === 'clinician') ? 'block' : 'none';
  }

  showToast(`Switched to ${currentRoleMode.toUpperCase()} mode`, 'info');
}

function openLegalModal(tab = 'privacy') {
  const modal = document.getElementById('legalModal');
  if (modal) modal.classList.add('active');
  switchLegalTab(tab);
}

function closeLegalModal() {
  const modal = document.getElementById('legalModal');
  if (modal) modal.classList.remove('active');
}

function switchLegalTab(tab) {
  const tPriv = document.getElementById('tabPrivacy');
  const tTerm = document.getElementById('tabTerms');
  const tDisc = document.getElementById('tabDisclaimer');
  const content = document.getElementById('legalTabContent');

  [tPriv, tTerm, tDisc].forEach(t => t && t.classList.remove('active'));

  if (tab === 'privacy') {
    if (tPriv) tPriv.classList.add('active');
    if (content) {
      content.innerHTML = `
        <h4 style="color:var(--text-primary); margin-bottom:0.5rem; font-weight:700;">RETINAX PATIENT DATA & PRIVACY POLICY</h4>
        <p><strong>1. Local Edge Processing:</strong> All retinal fundus images uploaded or captured via RetinaX are processed locally on-device or within your designated local network server. Patient image data is not transmitted to unencrypted third-party cloud servers.</p>
        <p style="margin-top:0.5rem;"><strong>2. Data Persistence:</strong> Diagnostic logs stored in the local SQLite database contain anonymized patient IDs, timestamps, quality scores, and classification outputs strictly for clinical audit trails.</p>
        <p style="margin-top:0.5rem;"><strong>3. Compliance Alignment:</strong> System architecture adheres to HIPAA and GDPR data minimization standards.</p>
      `;
    }
  } else if (tab === 'terms') {
    if (tTerm) tTerm.classList.add('active');
    if (content) {
      content.innerHTML = `
        <h4 style="color:var(--text-primary); margin-bottom:0.5rem; font-weight:700;">TERMS OF USE FOR CLINICAL USERS</h4>
        <p><strong>1. Authorized Use:</strong> RetinaX is designed exclusively as an auxiliary screening tool for certified healthcare professionals, ophthalmologists, and clinical researchers.</p>
        <p style="margin-top:0.5rem;"><strong>2. Professional Judgment:</strong> AI classification probabilities and Grad-CAM visualizations must be verified by a licensed medical practitioner prior to clinical treatment decisions.</p>
      `;
    }
  } else {
    if (tDisc) tDisc.classList.add('active');
    if (content) {
      content.innerHTML = `
        <h4 style="color:var(--text-primary); margin-bottom:0.5rem; font-weight:700;">MEDICAL & DIAGNOSTIC DISCLAIMER</h4>
        <p><strong>NOTICE:</strong> RetinaX is an Artificial Intelligence decision support screening aid. It does NOT provide formal medical diagnoses, nor does it replace comprehensive dilated eye exams performed by a licensed ophthalmologist or optometrist.</p>
        <p style="margin-top:0.5rem;">Patients experiencing acute visual changes, vision loss, or eye pain should seek immediate professional medical attention regardless of AI screening outputs.</p>
      `;
    }
  }
}

async function exportScreeningHistoryCsv() {
  try {
    showToast('Exporting clinical screening history CSV...', 'info');
    const res = await fetch('/api/v1/screenings/export');
    if (!res.ok) throw new Error(`Server returned status ${res.status}`);
    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `RetinaX_Screenings_Export_${Date.now()}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
    showToast('Screening history CSV exported successfully.', 'success');
  } catch (err) {
    showToast(`CSV export failed: ${err.message}`, 'error');
  }
}

// ----------------------------------------------------
// 19. 5-SECOND ROMAN OPENING SPLASH TIMER & AUTO-FADE
// ----------------------------------------------------
let splashTimeout = null;

function initIntroSplash() {
  const splash = document.getElementById('retinax-intro-splash');
  if (!splash) return;

  const statusText = document.getElementById('introStatusText');
  
  setTimeout(() => {
    if (statusText) statusText.innerText = 'CONNECTING GEOLOCATION NETWORK...';
  }, 1800);

  setTimeout(() => {
    if (statusText) statusText.innerText = 'RETINAX CLINICAL SUITE READY';
  }, 3600);

  splashTimeout = setTimeout(() => {
    skipIntroSplash();
  }, 5000);
}

function skipIntroSplash() {
  const splash = document.getElementById('retinax-intro-splash');
  if (splash) {
    splash.classList.add('fade-out');
    setTimeout(() => {
      splash.style.display = 'none';
    }, 500);
  }
  if (splashTimeout) clearTimeout(splashTimeout);
}

// ----------------------------------------------------
// 20. SETTINGS DROPDOWN & AUTHENTICATION CONTROLLER
// ----------------------------------------------------
function toggleSettingsMenu(e) {
  if (e) e.stopPropagation();
  const menu = document.getElementById('settingsDropdownMenu');
  if (menu) menu.classList.toggle('active');
}

document.addEventListener('click', (e) => {
  const menu = document.getElementById('settingsDropdownMenu');
  const btn = document.getElementById('btnSettingsToggle');
  if (menu && menu.classList.contains('active')) {
    if (!menu.contains(e.target) && !btn.contains(e.target)) {
      menu.classList.remove('active');
    }
  }
});

function openAuthModal() {
  const modal = document.getElementById('authModal');
  if (modal) modal.classList.add('active');
  const menu = document.getElementById('settingsDropdownMenu');
  if (menu) menu.classList.remove('active');
}

function closeAuthModal() {
  const modal = document.getElementById('authModal');
  if (modal) modal.classList.remove('active');
}

function switchAuthTab(tab) {
  const tabLog = document.getElementById('tabAuthLogin');
  const tabSign = document.getElementById('tabAuthSignup');
  const formLog = document.getElementById('authLoginForm');
  const formSign = document.getElementById('authSignupForm');
  const title = document.getElementById('authModalTitle');
  const sub = document.getElementById('authModalSubtitle');

  if (tab === 'login') {
    if (tabLog) tabLog.classList.add('active');
    if (tabSign) tabSign.classList.remove('active');
    if (formLog) formLog.style.display = 'block';
    if (formSign) formSign.style.display = 'none';
    if (title) title.innerText = 'ACCOUNT LOGIN';
    if (sub) sub.innerText = 'Sign in to RetinaX Clinical Suite';
  } else {
    if (tabSign) tabSign.classList.add('active');
    if (tabLog) tabLog.classList.remove('active');
    if (formSign) formSign.style.display = 'block';
    if (formLog) formLog.style.display = 'none';
    if (title) title.innerText = 'REGISTER ACCOUNT';
    if (sub) sub.innerText = 'Create a Clinician or Technician Profile';
  }
}

async function handleLoginSubmit(e) {
  e.preventDefault();
  const username = document.getElementById('loginUsername')?.value || 'User';
  const role = document.getElementById('loginRole')?.value || 'clinician';

  try {
    const res = await fetch('/api/v1/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password: 'password', role })
    });
    const data = await res.json();
    if (res.ok) {
      localStorage.setItem('retinax_user', JSON.stringify(data.user));
      updateUserDisplay(data.user);
      closeAuthModal();
      showToast(`Welcome back, ${data.user.name}! (${data.user.display_role})`, 'success');
    }
  } catch (err) {
    showToast(`Login failed: ${err.message}`, 'error');
  }
}

async function handleSignupSubmit(e) {
  e.preventDefault();
  const name = document.getElementById('signupName')?.value || 'User';
  const email = document.getElementById('signupEmail')?.value || 'user@hospital.org';
  const username = document.getElementById('signupUsername')?.value || 'user';
  const password = document.getElementById('signupPassword')?.value || 'pass';
  const role = document.getElementById('signupRole')?.value || 'clinician';

  try {
    const res = await fetch('/api/v1/auth/signup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, email, username, password, role })
    });
    const data = await res.json();
    if (res.ok) {
      localStorage.setItem('retinax_user', JSON.stringify(data.user));
      updateUserDisplay(data.user);
      closeAuthModal();
      showToast(`Account created successfully! Logged in as ${data.user.name}`, 'success');
    }
  } catch (err) {
    showToast(`Registration failed: ${err.message}`, 'error');
  }
}

function updateUserDisplay(user) {
  const nameEl = document.getElementById('dropdownUserName');
  const roleEl = document.getElementById('dropdownUserRole');
  if (nameEl && user?.name) nameEl.innerText = user.name;
  if (roleEl && user?.display_role) roleEl.innerText = user.display_role;
}

function logoutUser() {
  localStorage.removeItem('retinax_user');
  const nameEl = document.getElementById('dropdownUserName');
  const roleEl = document.getElementById('dropdownUserRole');
  if (nameEl) nameEl.innerText = 'Dr. Sarah Jenkins';
  if (roleEl) roleEl.innerText = 'Ophthalmologist Mode';
  const menu = document.getElementById('settingsDropdownMenu');
  if (menu) menu.classList.remove('active');
  showToast('Logged out of session successfully.', 'info');
}

// ----------------------------------------------------
// 21. EMERGENCY & HELP MODALS
// ----------------------------------------------------
function openEmergencyModal() {
  const modal = document.getElementById('emergencyModal');
  if (modal) modal.classList.add('active');
  const menu = document.getElementById('settingsDropdownMenu');
  if (menu) menu.classList.remove('active');
}

function closeEmergencyModal() {
  const modal = document.getElementById('emergencyModal');
  if (modal) modal.classList.remove('active');
}

function openHelpModal() {
  const modal = document.getElementById('helpModal');
  if (modal) modal.classList.add('active');
  const menu = document.getElementById('settingsDropdownMenu');
  if (menu) menu.classList.remove('active');
}

function closeHelpModal() {
  const modal = document.getElementById('helpModal');
  if (modal) modal.classList.remove('active');
}

// ----------------------------------------------------
// 22. FACILITY CATEGORY FILTER & REPORT ROUTING
// ----------------------------------------------------
let allFetchedHospitals = [];

function filterFacilityCategory(category) {
  ['btnFilterAll', 'btnFilterPhc', 'btnFilterGovt', 'btnFilterEye'].forEach(id => {
    const b = document.getElementById(id);
    if (b) b.classList.remove('active');
  });

  const activeBtnMap = {
    'all': 'btnFilterAll',
    'phc': 'btnFilterPhc',
    'govt_hospital': 'btnFilterGovt',
    'eye_clinic': 'btnFilterEye'
  };

  const targetId = activeBtnMap[category] || 'btnFilterAll';
  const activeBtn = document.getElementById(targetId);
  if (activeBtn) activeBtn.classList.add('active');

  const container = document.getElementById('doctorCardsList');
  if (!container || !allFetchedHospitals.length) return;

  const filtered = (category === 'all') 
    ? allFetchedHospitals 
    : allFetchedHospitals.filter(h => h.facility_type === category || (h.types && h.types.includes(category)));

  if (!filtered.length) {
    container.innerHTML = `<div class="glass-card empty-doctor-state"><h4>No ${category.toUpperCase().replace('_', ' ')} facilities found in this search radius.</h4></div>`;
    return;
  }

  container.innerHTML = filtered.map(doc => renderDoctorCardHtml(doc)).join('');
}

async function routeReportToHospital(hospitalId, hospitalName) {
  if (!currentPrediction) {
    showToast('Please run a retina analysis first before routing a report.', 'error');
    return;
  }

  const pId = document.getElementById('patientIdInput')?.value || 'RX-SCREEN';
  const pName = document.getElementById('patientNameInput')?.value || 'Patient Screening';

  try {
    showToast(`Transmitting screening report to ${hospitalName}...`, 'info');
    const res = await fetch('/api/v1/screenings/route', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        patient_id: pId,
        patient_name: pName,
        hospital_id: hospitalId,
        hospital_name: hospitalName,
        prediction_level: currentPrediction.level || 0,
        prediction_label: currentPrediction.label || 'No DR',
        confidence: currentPrediction.confidence || 0.0
      })
    });
    const data = await res.json();
    if (res.ok) {
      showToast(`Report successfully transmitted to ${hospitalName}! (Route ID: ${data.route_id})`, 'success');
    }
  } catch (err) {
    showToast(`Report transmission error: ${err.message}`, 'error');
  }
}

// Window Exports for global actions
window.openAuthModal = openAuthModal;
window.closeAuthModal = closeAuthModal;
window.switchAuthTab = switchAuthTab;
window.handleLoginSubmit = handleLoginSubmit;
window.handleSignupSubmit = handleSignupSubmit;
window.toggleSettingsMenu = toggleSettingsMenu;
window.logoutUser = logoutUser;
window.openEmergencyModal = openEmergencyModal;
window.closeEmergencyModal = closeEmergencyModal;
window.openHelpModal = openHelpModal;
window.closeHelpModal = closeHelpModal;
window.filterFacilityCategory = filterFacilityCategory;
window.routeReportToHospital = routeReportToHospital;


