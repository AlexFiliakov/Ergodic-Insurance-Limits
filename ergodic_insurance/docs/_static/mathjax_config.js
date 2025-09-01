/**
 * Custom MathJax configuration for Sphinx documentation
 * This ensures proper LaTeX rendering in theory pages
 */

// First, clear any existing MathJax configuration
if (window.MathJax) {
  window.MathJax = undefined;
}

window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true,
    processEnvironments: true,
    processRefs: true,
    tags: 'ams',
    tagSide: 'right',
    tagIndent: '0.8em',
    packages: {
      '[+]': ['ams', 'amssymb', 'amsmath', 'cases', 'boldsymbol']
    }
  },
  options: {
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
    ignoreHtmlClass: 'tex2jax_ignore|mathjax_ignore',
    processHtmlClass: 'tex2jax_process|mathjax_process|math|output_area'
  },
  chtml: {
    scale: 1,
    displayAlign: 'center',
    displayIndent: '0'
  },
  svg: {
    scale: 1,
    displayAlign: 'center',
    displayIndent: '0'
  },
  startup: {
    pageReady: () => {
      return MathJax.startup.defaultPageReady().then(() => {
        console.log('MathJax loaded and configured for Ergodic Insurance documentation');
        // Process any math divs that MyST created
        const mathDivs = document.querySelectorAll('div.math');
        mathDivs.forEach(div => {
          const mathText = div.textContent || div.innerText;
          if (mathText && !div.classList.contains('MathJax_Preview')) {
            // Replace the content with properly formatted math
            div.innerHTML = '\\[' + mathText.replace(/^\\\[/, '').replace(/\\\]$/, '') + '\\]';
            MathJax.typesetPromise([div]).catch((e) => console.error(e));
          }
        });
      });
    }
  }
};

// Load MathJax
(function () {
  const script = document.createElement('script');
  script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js';
  script.async = true;
  document.head.appendChild(script);
})();
