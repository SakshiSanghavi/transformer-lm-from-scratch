# NLP Visual Study Guide

This is a static site made from HTML, CSS, and JavaScript. It has no build step. The homepage links to five lecture revision pages and a Transformer implementation lab.

## Preview locally

Open `docs/index.html` in a browser. KaTeX and Google Fonts use CDN links, so those enhancements need an internet connection; the study text and diagrams are local.

## Publish with GitHub Pages

The repository workflow publishes the contents of `docs/` when a commit reaches the `develop` branch. In the repository settings, choose **Pages → Build and deployment → Source: GitHub Actions** once, then push the site changes. The workflow run reports the final Pages URL under its `github-pages` deployment environment.
