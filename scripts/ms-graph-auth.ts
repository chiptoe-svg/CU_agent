#!/usr/bin/env npx tsx
/**
 * Interactive OAuth2 auth flow for MS Graph (delegated permissions).
 * Opens a browser for login, captures the code via a local HTTP server,
 * exchanges it for tokens, and saves the refresh token to .env.
 */

import * as msal from "@azure/msal-node";
import * as http from "http";
import * as fs from "fs";
import * as path from "path";
import { execSync } from "child_process";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ENV_PATH = path.resolve(__dirname, "../.env");

function loadEnvVar(name: string): string {
  const content = fs.readFileSync(ENV_PATH, "utf-8");
  const match = content.match(new RegExp(`^${name}=(.+)$`, "m"));
  if (!match) throw new Error(`${name} not found in .env`);
  return match[1].trim();
}

const CLIENT_ID = loadEnvVar("MS365_MCP_CLIENT_ID");
const TENANT_ID = loadEnvVar("MS365_MCP_TENANT_ID");
const CLIENT_SECRET = loadEnvVar("MS365_MCP_CLIENT_SECRET");

const REDIRECT_URI = "http://localhost:3847/callback";

const SCOPES = [
  "Calendars.Read",
  "Calendars.ReadWrite",
  "Chat.Read",
  "Mail.Read",
  "Mail.ReadWrite",
  "Tasks.Read",
  "Tasks.ReadWrite",
  "User.Read",
  "offline_access", // needed to get a refresh token
];

const msalConfig: msal.Configuration = {
  auth: {
    clientId: CLIENT_ID,
    authority: `https://login.microsoftonline.com/${TENANT_ID}`,
    clientSecret: CLIENT_SECRET,
  },
};

const cca = new msal.ConfidentialClientApplication(msalConfig);

function saveToEnv(key: string, value: string) {
  let content = fs.readFileSync(ENV_PATH, "utf-8");
  const regex = new RegExp(`^${key}=.*$`, "m");
  if (regex.test(content)) {
    content = content.replace(regex, `${key}=${value}`);
  } else {
    content = content.trimEnd() + `\n${key}=${value}\n`;
  }
  fs.writeFileSync(ENV_PATH, content);
}

async function main() {
  // Generate the auth URL
  const authUrl = await cca.getAuthCodeUrl({
    scopes: SCOPES,
    redirectUri: REDIRECT_URI,
  });

  console.log("\n🔐 MS Graph Authentication\n");
  console.log("Opening browser for login...\n");

  // Start local server to catch the callback
  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url!, `http://localhost:3847`);

    if (url.pathname === "/callback") {
      const code = url.searchParams.get("code");
      const error = url.searchParams.get("error");

      if (error) {
        const desc = url.searchParams.get("error_description") || error;
        res.writeHead(400, { "Content-Type": "text/html" });
        res.end(`<h2>Auth failed</h2><pre>${desc}</pre>`);
        console.error("❌ Auth failed:", desc);
        server.close();
        process.exit(1);
      }

      if (!code) {
        res.writeHead(400, { "Content-Type": "text/html" });
        res.end("<h2>No authorization code received</h2>");
        server.close();
        process.exit(1);
      }

      try {
        const result = await cca.acquireTokenByCode({
          code,
          scopes: SCOPES,
          redirectUri: REDIRECT_URI,
        });

        // Save tokens to .env
        if (result.accessToken) {
          saveToEnv("MS365_MCP_ACCESS_TOKEN", result.accessToken);
        }

        // The refresh token isn't directly exposed by MSAL — check the cache
        const cache = cca.getTokenCache().serialize();
        const cacheObj = JSON.parse(cache);
        const refreshTokens = cacheObj.RefreshToken || {};
        const rtKey = Object.keys(refreshTokens)[0];
        if (rtKey) {
          const refreshToken = refreshTokens[rtKey].secret;
          saveToEnv("MS365_MCP_REFRESH_TOKEN", refreshToken);
          console.log("✅ Refresh token saved to .env");
        }

        // Save the token cache for MSAL to reuse
        const cachePath = path.resolve(__dirname, "../.ms-graph-cache.json");
        fs.writeFileSync(cachePath, cache);
        console.log("✅ Token cache saved to .ms-graph-cache.json");

        console.log(`✅ Access token obtained for: ${result.account?.username}`);
        console.log(`   Scopes: ${result.scopes.join(", ")}`);
        console.log(
          `   Expires: ${result.expiresOn?.toLocaleString() || "unknown"}`
        );

        res.writeHead(200, { "Content-Type": "text/html" });
        res.end(
          `<h2>✅ Authenticated!</h2><p>Signed in as <b>${result.account?.username}</b></p><p>You can close this window.</p>`
        );
      } catch (err: any) {
        console.error("❌ Token exchange failed:", err.message);
        res.writeHead(500, { "Content-Type": "text/html" });
        res.end(`<h2>Token exchange failed</h2><pre>${err.message}</pre>`);
      }

      setTimeout(() => {
        server.close();
        process.exit(0);
      }, 1000);
    }
  });

  server.listen(3847, () => {
    // Open browser
    try {
      execSync(`open "${authUrl}"`);
    } catch {
      console.log("Could not open browser. Visit this URL manually:\n");
      console.log(authUrl);
    }
  });
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
