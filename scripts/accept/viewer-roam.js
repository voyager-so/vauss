#!/usr/bin/env node
// The free-roam capsule acceptance replaces the legacy closed-box/ghost-escape checks.
// node viewer-roam.js [PAGE.html]: checks that exact page (an exported / patched index.html), else the template.
// Requires THREE_MODULE=/path/to/exported/viewer/vendor/three.module.js or npm three.
import('../../dev/free-roam/check-navigation.mjs');
