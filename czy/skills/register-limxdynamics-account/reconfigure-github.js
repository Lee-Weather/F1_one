// Reconfigure GitHub credentials for an existing LimX account (login + editGitInfo only).
'use strict';
const crypto = require('crypto');
const { DEFAULT_CREDENTIALS_FILE, readGithubCredentials, configureGithub } =
  require('./scripts/github-config');

const BASE = 'https://internal.limxdynamics.com/dev-api/api';
const EMAIL = 'limxmspwqm1g9hgfme@emalupe.com';
const PASSWORD = process.env.LIMX_PASSWORD || 'password123';

async function jsonFetch(url, options = {}) {
  const res = await fetch(url, options);
  const text = await res.text();
  let json;
  try { json = JSON.parse(text); } catch (_) { json = { raw: text }; }
  return { status: res.status, json };
}

(async () => {
  const creds = readGithubCredentials(DEFAULT_CREDENTIALS_FILE);
  const login = await jsonFetch(`${BASE}/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      username: EMAIL,
      password: crypto.createHash('md5').update(PASSWORD).digest('hex'),
      autoLogin: true
    })
  });
  console.log('login: HTTP ' + login.status + ' code=' + (login.json && login.json.code));
  const token = login.json && login.json.token;
  if (!token) throw new Error('登录失败（密码可能不是默认值）: ' + JSON.stringify(login.json).slice(0, 200));

  const result = await configureGithub(BASE, token, creds, jsonFetch);
  console.log('GITHUB_CONFIGURED=' + JSON.stringify(result));
})().catch((e) => { console.error('ERROR:', e.message); process.exit(1); });
