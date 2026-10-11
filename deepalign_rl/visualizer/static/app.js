// DeepAlign-RL Frontend Application Logic

let curatedProblems = [];

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  fetchStatus();
  fetchCuratedProblems();
  fetchBenchmarks();
  setupEventListeners();
});

// Tab Navigation
function initTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const targetId = tab.getAttribute("data-tab");
      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");

      document.querySelectorAll(".tab-panel").forEach(panel => {
        panel.classList.remove("active");
      });
      const activePanel = document.getElementById(`tab-${targetId}`);
      if (activePanel) {
        activePanel.classList.add("active");
      }
    });
  });
}

// Fetch Backend Status
async function fetchStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    const statusText = document.getElementById("device-status-text");
    if (statusText) {
      statusText.textContent = `${data.device} • Engine Active`;
    }
  } catch (err) {
    console.warn("Status fetch failed (API may be offline):", err);
    const statusText = document.getElementById("device-status-text");
    if (statusText) {
      statusText.textContent = "Engine Offline (Demo Mode)";
    }
  }
}

// Fetch Curated Problems
async function fetchCuratedProblems() {
  try {
    const res = await fetch("/api/problems");
    curatedProblems = await res.json();
    populateProblemDropdown(curatedProblems);
    if (curatedProblems.length > 0) {
      loadProblemIntoForm(curatedProblems[0]);
    }
  } catch (err) {
    console.warn("Problems fetch failed:", err);
  }
}

function populateProblemDropdown(problems) {
  const select = document.getElementById("sample-select");
  if (!select) return;

  select.innerHTML = '<option value="">-- Choose a Problem --</option>';
  problems.forEach((p, idx) => {
    const opt = document.createElement("option");
    opt.value = idx;
    opt.textContent = `[${p.category.toUpperCase()}] ${p.prompt.slice(0, 48)}...`;
    select.appendChild(opt);
  });

  select.addEventListener("change", (e) => {
    const idx = e.target.value;
    if (idx !== "" && problems[idx]) {
      loadProblemIntoForm(problems[idx]);
    }
  });
}

function loadProblemIntoForm(problem) {
  document.getElementById("prompt-input").value = problem.prompt;
  document.getElementById("completion-input").value = problem.target_solution;
  document.getElementById("gt-input").value = problem.ground_truth;
}

// Setup Event Listeners
function setupEventListeners() {
  const verifyBtn = document.getElementById("verify-run-btn");
  if (verifyBtn) {
    verifyBtn.addEventListener("click", runVerification);
  }

  const loadSampleBtn = document.getElementById("verify-load-sample-btn");
  if (loadSampleBtn) {
    loadSampleBtn.addEventListener("click", () => {
      if (curatedProblems.length > 0) {
        const randomProb = curatedProblems[Math.floor(Math.random() * curatedProblems.length)];
        loadProblemIntoForm(randomProb);
        runVerification();
      }
    });
  }

  const grpoBtn = document.getElementById("grpo-simulate-btn");
  if (grpoBtn) {
    grpoBtn.addEventListener("click", runGRPOSimulation);
  }
}

// Verifier Execution
async function runVerification() {
  const prompt = document.getElementById("prompt-input").value;
  const completion = document.getElementById("completion-input").value;
  const groundTruth = document.getElementById("gt-input").value;

  const badge = document.getElementById("reward-badge");
  badge.textContent = "Verifying...";
  badge.className = "badge badge-neutral";

  try {
    const res = await fetch("/api/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, completion, ground_truth: groundTruth }),
    });
    const data = await res.json();
    renderVerificationResults(data);
  } catch (err) {
    console.error("Verification error:", err);
    badge.textContent = "Verification Error";
    badge.className = "badge badge-failure";
  }
}

function renderVerificationResults(data) {
  const valComposite = document.getElementById("val-composite-reward");
  const valMath = document.getElementById("val-math-correct");
  const valFormat = document.getElementById("val-format-valid");
  const badge = document.getElementById("reward-badge");
  const container = document.getElementById("diagnostics-container");

  valComposite.textContent = data.total_reward.toFixed(2);
  valMath.textContent = data.is_correct ? "PASS" : "FAIL";
  valMath.className = data.is_correct ? "metric-value text-green" : "metric-value text-red";

  const fmtDetails = data.details?.components?.format_verifier || {};
  const isFormatValid = fmtDetails.format_valid;
  valFormat.textContent = isFormatValid ? "PASS" : "FAIL";
  valFormat.className = isFormatValid ? "metric-value text-green" : "metric-value text-red";

  if (data.is_correct && isFormatValid) {
    badge.textContent = "VERIFIED REASONING";
    badge.className = "badge badge-success";
  } else if (!data.is_correct) {
    badge.textContent = "INCORRECT ANSWER";
    badge.className = "badge badge-failure";
  } else {
    badge.textContent = "FORMATTING DEFECT";
    badge.className = "badge badge-failure";
  }

  const mathDetails = data.details?.components?.verifiable_math || {};

  container.innerHTML = `
    <div style="margin-bottom: 0.75rem;">
      <strong style="color: var(--accent-cyan);">SymPy Equivalence Check:</strong><br>
      • Extracted Candidate: <span style="color:#fff;">${mathDetails.extracted_pred || "None"}</span><br>
      • Target Canonical: <span style="color:#fff;">${mathDetails.ground_truth || "None"}</span><br>
      • SymPy Proof Status: <span style="color:${data.is_correct ? 'var(--accent-emerald)' : 'var(--accent-rose)'}; font-weight:700;">${mathDetails.explanation || "N/A"}</span>
    </div>
    <hr style="border:none; border-top:1px solid rgba(255,255,255,0.08); margin: 0.75rem 0;">
    <div>
      <strong style="color: var(--accent-purple);">Chain-of-Thought Structure:</strong><br>
      • &lt;think&gt; Tag Opened: ${fmtDetails.has_open_think ? '✔ Yes' : '✘ No'}<br>
      • &lt;/think&gt; Tag Closed: ${fmtDetails.has_close_think ? '✔ Yes' : '✘ No'}<br>
      • Reasoning Block Length: <span style="color:#fff;">${fmtDetails.think_length || 0} characters</span><br>
      • &lt;answer&gt; Tag Verified: ${fmtDetails.format_valid ? '✔ Yes' : '✘ No'}
    </div>
  `;
}

// GRPO Simulation
async function runGRPOSimulation() {
  const prompt = document.getElementById("grpo-prompt").value;
  const gt = document.getElementById("grpo-gt").value;

  // 4 candidate completions: 1 perfect, 1 correct but bad formatting, 1 arithmetic slip, 1 hallucinated
  const completions = [
    "<think>Subtract 15 from both sides: 3*x = 33. Divide both sides by 3: x = 11.</think>\n<answer>\\boxed{11}</answer>",
    "Subtract 15: 3*x = 33. x = 11. Answer is 11.",
    "<think>Subtract 15: 3*x = 35. Divide by 3: x = 11.666.</think>\n<answer>\\boxed{11.66}</answer>",
    "<think>Let's guess x = 10. 3*10 + 15 = 45, which is close enough to 48.</think>\n<answer>\\boxed{10}</answer>",
  ];

  try {
    const res = await fetch("/api/simulate-grpo", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, ground_truth: gt, completions }),
    });
    const data = await res.json();
    renderGRPOResults(data);
  } catch (err) {
    console.error("GRPO simulation error:", err);
  }
}

function renderGRPOResults(data) {
  const area = document.getElementById("grpo-rollouts-area");
  area.innerHTML = "";

  data.rollouts.forEach((rollout, idx) => {
    const card = document.createElement("div");
    const adv = rollout.advantage;
    const isPositive = adv > 0.05;
    const isNegative = adv < -0.05;

    card.className = `rollout-card ${isPositive ? 'positive' : (isNegative ? 'negative' : '')}`;

    const advClass = isPositive ? 'advantage-positive' : (isNegative ? 'advantage-negative' : 'advantage-zero');

    card.innerHTML = `
      <div class="rollout-header">
        <div>
          <strong>Rollout #${idx + 1}</strong>
          <span style="font-size:0.75rem; color:var(--text-muted); display:block;">
            Raw Reward: ${rollout.score.toFixed(2)}
          </span>
        </div>
        <div class="rollout-advantage-pill ${advClass}">
          Advantage: ${adv > 0 ? '+' : ''}${adv.toFixed(3)}
        </div>
      </div>
      <div class="rollout-text">${escapeHtml(rollout.completion)}</div>
      <div style="font-size: 0.75rem; color: var(--text-secondary); display:flex; justify-content:space-between;">
        <span>Math Equiv: <strong class="${rollout.is_correct ? 'text-green' : 'text-red'}">${rollout.is_correct ? 'TRUE' : 'FALSE'}</strong></span>
        <span>Group μ: ${data.mean_reward} | σ: ${data.std_reward}</span>
      </div>
    `;
    area.appendChild(card);
  });
}

// Benchmark Leaderboard Fetch
async function fetchBenchmarks() {
  try {
    const res = await fetch("/api/benchmark-stats");
    const data = await res.json();
    renderBenchmarkTable(data.benchmarks);
  } catch (err) {
    console.warn("Benchmark stats error:", err);
  }
}

function renderBenchmarkTable(benchmarks) {
  const tbody = document.getElementById("benchmark-table-body");
  if (!tbody) return;

  tbody.innerHTML = "";
  benchmarks.forEach(item => {
    const tr = document.createElement("tr");
    const isOurModel = item.method.includes("DeepAlign");
    if (isOurModel) tr.className = "highlight";

    tr.innerHTML = `
      <td><strong>${item.method}</strong></td>
      <td>${item.gsm8k_accuracy}%</td>
      <td>${item.math_accuracy}%</td>
      <td>${item.format_compliance}%</td>
      <td>${item.avg_reasoning_tokens}</td>
      <td>${item.critic_overhead_vram}</td>
      <td><span class="badge ${isOurModel ? 'badge-success' : 'badge-neutral'}">${isOurModel ? 'High (GRPO)' : 'Standard'}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}
