// MathJax configuration for the arithmatex extension (generic mode), which
// wraps inline math in \( ... \) and display math in \[ ... \] inside
// elements with class "arithmatex". Only those elements are typeset.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

// Instant navigation swaps the page content without a full reload, so
// typeset again on every page change.
document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
