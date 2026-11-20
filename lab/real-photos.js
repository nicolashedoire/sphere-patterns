(() => {
  "use strict";

  const modes = {
    photo: ["Camera image", "Inspect the original frame."],
    rgbd: ["RGB-D predictions", "The detector searched the whole color and depth frame for the target part."],
    wave: ["Returning-reference check", "The returning wave reference checked regions proposed by the RGB-D detector."],
    components: ["Depth components", "The detector cuts the measured depth at different heights. Each proposed region is checked against the target and similar parts."],
  };
  const resultStyles = {
    hit: ["match", "Match"],
    "false-alarm": ["false-alarm", "False alarm"],
    ignored: ["ignored", "Excluded from scoring"],
  };
  const byId = id => document.getElementById(id);
  const status = byId("photo-status");
  const photo = byId("scene-photo");
  const boxes = byId("boxes");
  const annotations = byId("annotations");
  const resultList = byId("results");
  const legend = byId("legend");
  const sceneNav = byId("scenes");
  const asset = name => "photo-data/" + name;
  const pct = value => Math.round(value * 100) + "%";
  const alarmText = count => `${count} false ${count === 1 ? "alarm" : "alarms"}`;
  const svgNS = "http://www.w3.org/2000/svg";
  const componentTrial = new URLSearchParams(location.search).get("trial") === "components";
  let benchmark;
  let selectedScene = 0;

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text) node.textContent = text;
    return node;
  }

  function svg(tag, attributes) {
    const node = document.createElementNS(svgNS, tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    return node;
  }

  function drawBox(scene, box, type, label) {
    const [x, y, width, height] = box;
    const group = svg("g", { class: type });
    group.append(svg("rect", { x, y, width, height }));
    const text = svg("text", {
      x: Math.min(scene.width - 31, x + 3),
      y: y > 23 ? y - 5 : y + 22,
    });
    text.textContent = label;
    group.append(text);
    boxes.append(group);
  }

  function listDetection(hit, index, reveal) {
    const [style, label] = reveal && resultStyles[hit.result] ? resultStyles[hit.result] : ["", "Accepted region"];
    const row = element("li");
    const marker = element("span", "photo-dot photo-mark " + style);
    marker.setAttribute("aria-hidden", "true");
    const details = reveal && hit.result === "hit"
      ? "Overlap with annotated box " + pct(hit.overlap)
      : hit.reference?.startsWith("train-") ? "Training view " + hit.reference : "Candidate check";
    row.append(marker,
      element("span", "", (index + 1) + ". " + label),
      element("span", "photo-value", "Score " + hit.score.toFixed(3)),
      element("span", "photo-description", details));
    resultList.append(row);
    return style;
  }

  function describe(scene, predictions, mode, reveal) {
    const accepted = predictions.length === 1 ? "1 accepted region" : predictions.length + " accepted regions";
    if (!reveal) return mode === "photo" ? "Camera image" : accepted;
    const groundTruth = scene.targets.length === 1 ? "1 annotated target" : scene.targets.length + " annotated targets";
    if (mode === "photo") return groundTruth;
    const matched = predictions.filter(item => item.result === "hit").length;
    const falseAlarms = predictions.filter(item => item.result === "false-alarm").length;
    const missed = scene.targets.length - matched;
    return accepted + " · " + matched + " matched · " + alarmText(falseAlarms) + " · " + missed + " missed";
  }

  function render() {
    if (!benchmark) return;
    const scene = benchmark.scenes[selectedScene];
    const mode = document.querySelector('input[name="layer"]:checked').value;
    const reveal = annotations.checked;
    const predictions = scene.detections[mode] || [];
    const source = asset(scene.image);
    if (photo.getAttribute("src") !== source) photo.src = source;
    photo.alt = "T-LESS camera image, scene " + scene.scene + ", frame " + scene.frame + ".";
    photo.width = scene.width;
    photo.height = scene.height;
    boxes.setAttribute("viewBox", "0 0 " + scene.width + " " + scene.height);
    byId("scene-caption").textContent = "Scene " + scene.scene + " · Frame " + scene.frame;
    byId("readout-title").textContent = modes[mode][0];
    byId("mode-detail").textContent = modes[mode][1];
    const totals = benchmark.totals?.[mode];
    const trialSummary = byId("trial-summary");
    trialSummary.hidden = !totals;
    trialSummary.textContent = totals ? `${totals.hits}/${totals.eligibleInstances} matched · ${alarmText(totals.falseAlarms)} across ${benchmark.photoCount} views` : "";
    byId("summary").textContent = describe(scene, predictions, mode, reveal);
    const note = byId("annotation-note");
    note.hidden = !reveal;
    note.textContent = "Scoring uses targets that are at least " + pct(benchmark.visibilityThreshold)
      + " visible and requires " + pct(benchmark.overlapThreshold) + " bounding-box overlap (IoU).";
    boxes.replaceChildren();
    legend.replaceChildren();
    resultList.replaceChildren();
    for (const [index, prediction] of predictions.entries()) {
      const style = listDetection(prediction, index, reveal);
      drawBox(scene, prediction.box, "detection " + style, String(index + 1));
    }
    if (predictions.length && !reveal) legend.append(element("li", "photo-mark", "Accepted region"));
    if (reveal) {
      for (const [index, target] of scene.targets.entries()) {
        drawBox(scene, target.box, "annotation", "T" + (index + 1));
      }
      const present = new Set(predictions.map(item => item.result));
      for (const [result, [style, label]] of Object.entries(resultStyles)) {
        if (present.has(result)) legend.append(element("li", "photo-mark " + style, label));
      }
      if (scene.targets.length) legend.append(element("li", "photo-mark annotation", "Annotated target"));
    }
    for (const [index, button] of [...sceneNav.children].entries()) {
      button.setAttribute("aria-pressed", String(index === selectedScene));
    }
  }

  async function load() {
    const response = await fetch(asset(componentTrial ? "benchmark-components.json" : "benchmark.json"));
    if (!response.ok) throw new Error("Could not load the benchmark.");
    benchmark = await response.json();
    if (componentTrial) {
      byId("components-mode").hidden = false;
      document.querySelector('input[value="components"]').checked = true;
      annotations.checked = true;
      byId("trial-link").href = "real-photos.html";
      byId("trial-link").textContent = "First photo trial";
      byId("photo-intro").textContent = `Compare the three detectors on ${benchmark.scenes.length} examples from ${benchmark.photoCount} held-out camera views.`;
      byId("measurement-link").href = "https://github.com/nicolashedoire/sphere-patterns/blob/main/outputs/sphere-tless-component-measurements.json";
    }
    for (const [index, scene] of benchmark.scenes.entries()) {
      const button = element("button");
      const thumb = element("img");
      thumb.src = asset(scene.image);
      thumb.alt = "";
      button.type = "button";
      button.append(thumb, element("strong", "", "Scene " + scene.scene + " · " + scene.frame));
      button.addEventListener("click", () => {
        selectedScene = index;
        render();
      });
      sceneNav.append(button);
    }
    byId("layers").addEventListener("change", render);
    annotations.addEventListener("change", render);
    render();
    byId("photo-content").hidden = false;
    status.hidden = true;
  }

  load().catch(error => {
    status.textContent = error.message;
  });
})();
