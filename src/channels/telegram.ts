import fs from 'fs';
import https from 'https';
import path from 'path';

import { Api, Bot } from 'grammy';

import {
  getCurrentAuthMode,
  hasValidOAuthCredentials,
  switchAuthMode,
} from '../auth-switch.js';
import {
  ASSISTANT_NAME,
  AVAILABLE_MODELS,
  DATA_DIR,
  DEFAULT_MODEL,
  DEFAULT_RUNTIME,
  LITELLM_URL,
  OMLX_URL,
  TRIGGER_PATTERN,
} from '../config.js';
import { getRegisteredGroup, setRegisteredGroup } from '../db.js';
import { readEnvFile } from '../env.js';
import { resolveGroupFolderPath } from '../group-folder.js';
import { logger } from '../logger.js';
import { transcribeAudio } from '../transcription.js';
import { registerChannel, ChannelOpts } from './registry.js';
import {
  Channel,
  OnChatMetadata,
  OnInboundMessage,
  RegisteredGroup,
} from '../types.js';

export interface TelegramChannelOpts {
  onMessage: OnInboundMessage;
  onChatMetadata: OnChatMetadata;
  registeredGroups: () => Record<string, RegisteredGroup>;
}

type TelegramMediaCtx = {
  chat: { id: number; type: string; title?: string };
  from?: { id?: number; first_name?: string; username?: string };
  message: {
    date: number;
    message_id: number;
    caption?: string;
  };
};

/**
 * Send a message with Telegram Markdown parse mode, falling back to plain text.
 * Claude's output naturally matches Telegram's Markdown v1 format:
 *   *bold*, _italic_, `code`, ```code blocks```, [links](url)
 */
async function sendTelegramMessage(
  api: { sendMessage: Api['sendMessage'] },
  chatId: string | number,
  text: string,
  options: { message_thread_id?: number } = {},
): Promise<void> {
  try {
    await api.sendMessage(chatId, text, {
      ...options,
      parse_mode: 'Markdown',
    });
  } catch (err) {
    // Fallback: send as plain text if Markdown parsing fails
    logger.debug({ err }, 'Markdown send failed, falling back to plain text');
    await api.sendMessage(chatId, text, options);
  }
}

export class TelegramChannel implements Channel {
  name = 'telegram';

  private bot: Bot | null = null;
  private opts: TelegramChannelOpts;
  private botToken: string;

  constructor(botToken: string, opts: TelegramChannelOpts) {
    this.botToken = botToken;
    this.opts = opts;
  }

  /**
   * Download a Telegram file to the group's attachments directory.
   * Returns the container-relative path (e.g. /workspace/group/attachments/photo_123.jpg)
   * or null if the download fails.
   */
  private async downloadFile(
    fileId: string,
    groupFolder: string,
    filename: string,
  ): Promise<string | null> {
    if (!this.bot) return null;

    try {
      const file = await this.bot.api.getFile(fileId);
      if (!file.file_path) {
        logger.warn({ fileId }, 'Telegram getFile returned no file_path');
        return null;
      }

      const groupDir = resolveGroupFolderPath(groupFolder);
      const attachDir = path.join(groupDir, 'attachments');
      fs.mkdirSync(attachDir, { recursive: true });

      // Sanitize filename and add extension from Telegram's file_path if missing
      const tgExt = path.extname(file.file_path);
      const localExt = path.extname(filename);
      const safeName = filename.replace(/[^a-zA-Z0-9._-]/g, '_');
      const finalName = localExt ? safeName : `${safeName}${tgExt}`;
      const destPath = path.join(attachDir, finalName);

      const fileUrl = `https://api.telegram.org/file/bot${this.botToken}/${file.file_path}`;
      const resp = await fetch(fileUrl);
      if (!resp.ok) {
        logger.warn(
          { fileId, status: resp.status },
          'Telegram file download failed',
        );
        return null;
      }

      const buffer = Buffer.from(await resp.arrayBuffer());
      fs.writeFileSync(destPath, buffer);

      logger.info({ fileId, dest: destPath }, 'Telegram file downloaded');
      return `/workspace/group/attachments/${finalName}`;
    } catch (err) {
      logger.error({ fileId, err }, 'Failed to download Telegram file');
      return null;
    }
  }

  async connect(): Promise<void> {
    this.bot = new Bot(this.botToken, {
      client: {
        baseFetchConfig: { agent: https.globalAgent, compress: true },
      },
    });

    // Command to get chat ID (useful for registration)
    this.bot.command('chatid', (ctx) => {
      const chatId = ctx.chat.id;
      const chatType = ctx.chat.type;
      const chatName =
        chatType === 'private'
          ? ctx.from?.first_name || 'Private'
          : ('title' in ctx.chat ? ctx.chat.title : undefined) || 'Unknown';

      ctx.reply(
        `Chat ID: \`tg:${chatId}\`\nName: ${chatName}\nType: ${chatType}`,
        { parse_mode: 'Markdown' },
      );
    });

    // Command to check bot status
    this.bot.command('ping', (ctx) => {
      ctx.reply(`${ASSISTANT_NAME} is online.`);
    });

    // Command to view or switch auth mode (runtime-aware)
    this.bot.command('auth', async (ctx) => {
      const chatJid = `tg:${ctx.chat.id}`;
      const group = getRegisteredGroup(chatJid);
      const runtime = group?.containerConfig?.runtime || DEFAULT_RUNTIME;
      const args = ctx.message?.text?.split(/\s+/).slice(1) || [];

      if (runtime === 'codex') {
        const envSecrets = readEnvFile(['OPENAI_API_KEY']);
        const hasApiKey = !!envSecrets.OPENAI_API_KEY;
        const codexAuthFile = path.join(
          process.env.HOME || '/home/node',
          '.codex',
          'auth.json',
        );
        const hasSubscription = fs.existsSync(codexAuthFile);
        const currentModel = group?.containerConfig?.model || DEFAULT_MODEL;

        let authMode: string;
        if (hasSubscription && hasApiKey) {
          authMode = 'Subscription (+ API key fallback)';
        } else if (hasSubscription) {
          authMode = 'Subscription';
        } else if (hasApiKey) {
          authMode = 'API Key';
        } else {
          authMode = 'not configured';
        }

        if (args.length === 0) {
          ctx.reply(
            `Runtime: *OpenAI (Codex)*\nModel: ${currentModel}\nAuth: ${authMode}\n\nSwitch with:\n\`/auth subscription\` — use ChatGPT subscription\n\`/auth api\` — use API key from .env`,
            { parse_mode: 'Markdown' },
          );
          return;
        }

        const target = args[0].toLowerCase();
        if (target === 'subscription') {
          if (hasSubscription) {
            ctx.reply('Subscription auth already configured.');
          } else {
            ctx.reply(
              'Run `codex auth login` on the server, then send `/auth` again to verify.',
              { parse_mode: 'Markdown' },
            );
          }
          return;
        }
        if (target === 'api') {
          if (hasApiKey) {
            ctx.reply('API key already configured in .env');
          } else {
            ctx.reply(
              'Add `OPENAI_API_KEY=sk-...` to .env on the server, then restart the service.',
              { parse_mode: 'Markdown' },
            );
          }
          return;
        }

        ctx.reply('Usage: `/auth subscription` or `/auth api`', {
          parse_mode: 'Markdown',
        });
        return;
      }

      if (runtime === 'gemini') {
        const envSecrets = readEnvFile(['GEMINI_API_KEY', 'GOOGLE_API_KEY']);
        const hasKey = !!(
          envSecrets.GEMINI_API_KEY || envSecrets.GOOGLE_API_KEY
        );
        const currentModel =
          group?.containerConfig?.model || 'gemini-2.5-flash';
        ctx.reply(
          `Runtime: *Gemini*\nModel: ${currentModel}\nAPI key: ${hasKey ? 'configured' : 'not configured (free tier available)'}`,
          { parse_mode: 'Markdown' },
        );
        return;
      }

      // Claude runtime: show auth mode with switching options
      const current = getCurrentAuthMode();

      if (args.length === 0) {
        const label =
          current === 'api-key' ? 'API Key' : 'OAuth (Subscription)';
        let credStatus: string;
        if (current === 'api-key') {
          credStatus = 'active (API key in .env)';
        } else {
          const envSecrets = readEnvFile([
            'CLAUDE_CODE_OAUTH_TOKEN',
            'ANTHROPIC_AUTH_TOKEN',
          ]);
          const hasEnvToken = !!(
            envSecrets.CLAUDE_CODE_OAUTH_TOKEN ||
            envSecrets.ANTHROPIC_AUTH_TOKEN
          );
          const hasCliCreds = hasValidOAuthCredentials();
          if (hasEnvToken && hasCliCreds) {
            credStatus = 'active (.env token + CLI auto-refresh)';
          } else if (hasEnvToken) {
            credStatus = 'active (.env token)';
          } else if (hasCliCreds) {
            credStatus = 'active (CLI auto-refresh)';
          } else {
            credStatus = 'no credentials found';
          }
        }
        ctx.reply(
          `Runtime: *Claude*\nAuth mode: *${label}*\nCredentials: ${credStatus}\n\nSwitch with:\n\`/auth api\` — use API key\n\`/auth oauth\` — use subscription`,
          { parse_mode: 'Markdown' },
        );
        return;
      }

      const target = args[0].toLowerCase();
      if (target !== 'api' && target !== 'oauth') {
        ctx.reply('Usage: `/auth api` or `/auth oauth`', {
          parse_mode: 'Markdown',
        });
        return;
      }

      const newMode = target === 'api' ? 'api-key' : 'oauth';
      if (newMode === current) {
        const label = current === 'api-key' ? 'API Key' : 'OAuth';
        ctx.reply(`Already using ${label}.`);
        return;
      }

      if (newMode === 'oauth' && !hasValidOAuthCredentials()) {
        ctx.reply(
          'No valid OAuth credentials found.\nRun `claude login` on the server first, then try again.',
          { parse_mode: 'Markdown' },
        );
        return;
      }

      switchAuthMode(newMode as 'api-key' | 'oauth');
      const label = newMode === 'api-key' ? 'API Key' : 'OAuth (Subscription)';
      ctx.reply(`Switching to *${label}*...`, {
        parse_mode: 'Markdown',
      });

      const flagPath = path.join(DATA_DIR, 'auth-switch-pending.json');
      fs.mkdirSync(DATA_DIR, { recursive: true });
      fs.writeFileSync(
        flagPath,
        JSON.stringify({ chatId: ctx.chat.id, mode: newMode }),
      );

      logger.info({ newMode }, 'Auth mode changed, restarting service');
      setTimeout(() => {
        process.exit(0);
      }, 1000);
    });

    // Command to view or switch model
    this.bot.command('model', async (ctx) => {
      const chatJid = `tg:${ctx.chat.id}`;
      const group = getRegisteredGroup(chatJid);
      if (!group) {
        ctx.reply('This chat is not registered.');
        return;
      }

      const runtime = group.containerConfig?.runtime || DEFAULT_RUNTIME;
      const currentModel = group.containerConfig?.model || DEFAULT_MODEL;
      const args = ctx.message?.text?.split(/\s+/).slice(1) || [];

      if (args.length === 0) {
        const runtimeModels = AVAILABLE_MODELS[runtime] || [];
        // Local and custom models only work via Codex's baseUrl routing
        const localModels =
          runtime === 'codex' ? AVAILABLE_MODELS['local'] || [] : [];
        const customModels =
          runtime === 'codex' && LITELLM_URL
            ? AVAILABLE_MODELS['custom'] || []
            : [];
        const baseUrl = group.containerConfig?.baseUrl;

        let reply = `Model: *${currentModel}*\nRuntime: ${runtime}`;
        if (baseUrl) {
          reply += `\nEndpoint: ${baseUrl}`;
        }

        if (runtimeModels.length > 0) {
          reply +=
            '\n\n*' +
            runtime +
            ' models:*\n' +
            runtimeModels
              .map(
                (m) =>
                  `${m.id === currentModel ? '→' : '  '} \`${m.id}\` — ${m.name}`,
              )
              .join('\n');
        }

        if (localModels.length > 0) {
          reply +=
            '\n\n*Local models (OMLX):*\n' +
            localModels
              .map(
                (m) =>
                  `${m.id === currentModel ? '→' : '  '} \`${m.id}\` — ${m.name}`,
              )
              .join('\n');
        }

        if (customModels.length > 0) {
          reply +=
            '\n\n*Custom models (LiteLLM):*\n' +
            customModels
              .map(
                (m) =>
                  `${m.id === currentModel ? '→' : '  '} \`${m.id}\` — ${m.name}`,
              )
              .join('\n');
        }

        reply += '\n\nSwitch with: `/model <name>`';
        ctx.reply(reply, { parse_mode: 'Markdown' });
        return;
      }

      const target = args[0];
      if (target === currentModel) {
        ctx.reply(`Already using ${target}.`);
        return;
      }

      // Determine routing: local (OMLX direct), custom (LiteLLM), or standard (no proxy)
      const isLocalModel = (AVAILABLE_MODELS['local'] || []).some(
        (m) => m.id === target,
      );
      const isCustomModel = (AVAILABLE_MODELS['custom'] || []).some(
        (m) => m.id === target,
      );

      // Local and custom models only work via Codex's baseUrl routing
      if ((isLocalModel || isCustomModel) && runtime !== 'codex') {
        ctx.reply(
          `Local/custom models require the Codex runtime. This group uses *${runtime}*.\nSwitch runtime first, or pick a ${runtime} model.`,
          { parse_mode: 'Markdown' },
        );
        return;
      }

      const config = group.containerConfig || {};
      config.model = target;

      if (isLocalModel) {
        config.baseUrl = OMLX_URL;
      } else if (isCustomModel) {
        if (!LITELLM_URL) {
          ctx.reply(
            'Custom models require LiteLLM. Set `LITELLM_URL` in .env and start LiteLLM.',
            { parse_mode: 'Markdown' },
          );
          return;
        }
        config.baseUrl = LITELLM_URL;
      } else {
        delete config.baseUrl;
      }

      setRegisteredGroup(chatJid, {
        ...group,
        containerConfig: config,
      });

      const via = isLocalModel
        ? ' (local via OMLX)'
        : isCustomModel
          ? ' (via LiteLLM)'
          : '';
      ctx.reply(`Model switched to *${target}*${via}`, {
        parse_mode: 'Markdown',
      });
    });

    // Telegram bot commands handled above — skip them in the general handler
    // so they don't also get stored as messages. All other /commands flow through.
    const TELEGRAM_BOT_COMMANDS = new Set(['chatid', 'ping', 'auth', 'model']);

    this.bot.on('message:text', async (ctx) => {
      if (ctx.message.text.startsWith('/')) {
        const cmd = ctx.message.text.slice(1).split(/[\s@]/)[0].toLowerCase();
        if (TELEGRAM_BOT_COMMANDS.has(cmd)) return;
      }

      const chatJid = `tg:${ctx.chat.id}`;
      let content = ctx.message.text;
      const timestamp = new Date(ctx.message.date * 1000).toISOString();
      const senderName =
        ctx.from?.first_name ||
        ctx.from?.username ||
        ctx.from?.id.toString() ||
        'Unknown';
      const sender = ctx.from?.id.toString() || '';
      const msgId = ctx.message.message_id.toString();
      const threadId = ctx.message.message_thread_id;

      const replyTo = ctx.message.reply_to_message;
      const replyToMessageId = replyTo?.message_id?.toString();
      const replyToContent = replyTo?.text || replyTo?.caption;
      const replyToSenderName = replyTo
        ? replyTo.from?.first_name ||
          replyTo.from?.username ||
          replyTo.from?.id?.toString() ||
          'Unknown'
        : undefined;

      // Determine chat name
      const chatName =
        ctx.chat.type === 'private'
          ? senderName
          : ('title' in ctx.chat ? ctx.chat.title : undefined) || chatJid;

      // Translate Telegram @bot_username mentions into TRIGGER_PATTERN format.
      // Telegram @mentions (e.g., @andy_ai_bot) won't match TRIGGER_PATTERN
      // (e.g., ^@Andy\b), so we prepend the trigger when the bot is @mentioned.
      const botUsername = ctx.me?.username?.toLowerCase();
      if (botUsername) {
        const entities = ctx.message.entities || [];
        const isBotMentioned = entities.some((entity) => {
          if (entity.type === 'mention') {
            const mentionText = content
              .substring(entity.offset, entity.offset + entity.length)
              .toLowerCase();
            return mentionText === `@${botUsername}`;
          }
          return false;
        });
        if (isBotMentioned && !TRIGGER_PATTERN.test(content)) {
          content = `@${ASSISTANT_NAME} ${content}`;
        }
      }

      // Store chat metadata for discovery
      const isGroup =
        ctx.chat.type === 'group' || ctx.chat.type === 'supergroup';
      this.opts.onChatMetadata(
        chatJid,
        timestamp,
        chatName,
        'telegram',
        isGroup,
      );

      // Only deliver full message for registered groups
      const group = this.opts.registeredGroups()[chatJid];
      if (!group) {
        logger.debug(
          { chatJid, chatName },
          'Message from unregistered Telegram chat',
        );
        return;
      }

      // Deliver message — startMessageLoop() will pick it up
      this.opts.onMessage(chatJid, {
        id: msgId,
        chat_jid: chatJid,
        sender,
        sender_name: senderName,
        content,
        timestamp,
        is_from_me: false,
        thread_id: threadId ? threadId.toString() : undefined,
        reply_to_message_id: replyToMessageId,
        reply_to_message_content: replyToContent,
        reply_to_sender_name: replyToSenderName,
      });

      logger.info(
        { chatJid, chatName, sender: senderName },
        'Telegram message stored',
      );
    });

    // Handle non-text messages: download files when possible, fall back to placeholders.
    const storeMedia = (
      ctx: TelegramMediaCtx,
      placeholder: string,
      opts?: { fileId?: string; filename?: string },
    ) => {
      const chatJid = `tg:${ctx.chat.id}`;
      const group = this.opts.registeredGroups()[chatJid];
      if (!group) return;

      const timestamp = new Date(ctx.message.date * 1000).toISOString();
      const senderName =
        ctx.from?.first_name ||
        ctx.from?.username ||
        ctx.from?.id?.toString() ||
        'Unknown';
      const caption = ctx.message.caption ? ` ${ctx.message.caption}` : '';

      const isGroup =
        ctx.chat.type === 'group' || ctx.chat.type === 'supergroup';
      this.opts.onChatMetadata(
        chatJid,
        timestamp,
        undefined,
        'telegram',
        isGroup,
      );

      const deliver = (content: string) => {
        this.opts.onMessage(chatJid, {
          id: ctx.message.message_id.toString(),
          chat_jid: chatJid,
          sender: ctx.from?.id?.toString() || '',
          sender_name: senderName,
          content,
          timestamp,
          is_from_me: false,
        });
      };

      // If we have a file_id, attempt to download; deliver asynchronously
      if (opts?.fileId) {
        const msgId = ctx.message.message_id.toString();
        const filename =
          opts.filename ||
          `${placeholder.replace(/[[\] ]/g, '').toLowerCase()}_${msgId}`;
        this.downloadFile(opts.fileId, group.folder, filename).then(
          (filePath) => {
            if (filePath) {
              deliver(`${placeholder} (${filePath})${caption}`);
            } else {
              deliver(`${placeholder}${caption}`);
            }
          },
        );
        return;
      }

      deliver(`${placeholder}${caption}`);
    };

    this.bot.on('message:photo', (ctx) => {
      // Telegram sends multiple sizes; last is largest
      const photos = ctx.message.photo;
      const largest = photos?.[photos.length - 1];
      storeMedia(ctx, '[Photo]', {
        fileId: largest?.file_id,
        filename: `photo_${ctx.message.message_id}`,
      });
    });
    this.bot.on('message:video', (ctx) => {
      storeMedia(ctx, '[Video]', {
        fileId: ctx.message.video?.file_id,
        filename: `video_${ctx.message.message_id}`,
      });
    });
    this.bot.on('message:voice', (ctx) => {
      const chatJid = `tg:${ctx.chat.id}`;
      const group = this.opts.registeredGroups()[chatJid];
      if (!group) return;

      const fileId = ctx.message.voice?.file_id;
      if (!fileId) {
        storeMedia(ctx, '[Voice message]');
        return;
      }

      const filename = `voice_${ctx.message.message_id}`;
      const timestamp = new Date(ctx.message.date * 1000).toISOString();
      const senderName =
        ctx.from?.first_name ||
        ctx.from?.username ||
        ctx.from?.id?.toString() ||
        'Unknown';
      const caption = ctx.message.caption ? ` ${ctx.message.caption}` : '';
      const isGroup =
        ctx.chat.type === 'group' || ctx.chat.type === 'supergroup';

      this.opts.onChatMetadata(
        chatJid,
        timestamp,
        undefined,
        'telegram',
        isGroup,
      );

      this.downloadFile(fileId, group.folder, filename).then(
        async (filePath) => {
          let content: string;
          if (filePath) {
            // Resolve the actual host path for transcription
            const groupDir = resolveGroupFolderPath(group.folder);
            const localFile = path.join(
              groupDir,
              'attachments',
              path.basename(filePath),
            );
            const transcript = await transcribeAudio(localFile);
            if (transcript) {
              content = `[Voice: ${transcript}]${caption}`;
            } else {
              content = `[Voice message] (${filePath})${caption}`;
            }
          } else {
            content = `[Voice message]${caption}`;
          }

          this.opts.onMessage(chatJid, {
            id: ctx.message.message_id.toString(),
            chat_jid: chatJid,
            sender: ctx.from?.id?.toString() || '',
            sender_name: senderName,
            content,
            timestamp,
            is_from_me: false,
          });
        },
      );
    });
    this.bot.on('message:audio', (ctx) => {
      const name =
        ctx.message.audio?.file_name || `audio_${ctx.message.message_id}`;
      storeMedia(ctx, '[Audio]', {
        fileId: ctx.message.audio?.file_id,
        filename: name,
      });
    });
    this.bot.on('message:document', (ctx) => {
      const name = ctx.message.document?.file_name || 'file';
      storeMedia(ctx, `[Document: ${name}]`, {
        fileId: ctx.message.document?.file_id,
        filename: name,
      });
    });
    this.bot.on('message:sticker', (ctx) => {
      const emoji = ctx.message.sticker?.emoji || '';
      storeMedia(ctx, `[Sticker ${emoji}]`);
    });
    this.bot.on('message:location', (ctx) => storeMedia(ctx, '[Location]'));
    this.bot.on('message:contact', (ctx) => storeMedia(ctx, '[Contact]'));

    // Handle errors gracefully
    this.bot.catch((err) => {
      logger.error({ err: err.message }, 'Telegram bot error');
    });

    // Start polling — returns a Promise that resolves when started
    return new Promise<void>((resolve) => {
      this.bot!.start({
        onStart: async (botInfo) => {
          logger.info(
            { username: botInfo.username, id: botInfo.id },
            'Telegram bot connected',
          );
          console.log(`\n  Telegram bot: @${botInfo.username}`);
          console.log(
            `  Send /chatid to the bot to get a chat's registration ID\n`,
          );

          // Send ready message if auth was just switched
          try {
            const flagPath = path.join(DATA_DIR, 'auth-switch-pending.json');
            if (fs.existsSync(flagPath)) {
              const flag = JSON.parse(fs.readFileSync(flagPath, 'utf-8'));
              fs.unlinkSync(flagPath);
              const mode = getCurrentAuthMode();
              const label =
                mode === 'api-key' ? 'API Key' : 'OAuth (Subscription)';
              await this.bot!.api.sendMessage(
                flag.chatId,
                `${ASSISTANT_NAME} is back online.\nAuth mode: *${label}*`,
                { parse_mode: 'Markdown' },
              );
            }
          } catch (err) {
            logger.debug({ err }, 'Failed to send auth switch notification');
          }

          resolve();
        },
      });
    });
  }

  async sendMessage(
    jid: string,
    text: string,
    threadId?: string,
  ): Promise<void> {
    if (!this.bot) {
      logger.warn('Telegram bot not initialized');
      return;
    }

    try {
      const numericId = jid.replace(/^tg:/, '');
      const options = threadId
        ? { message_thread_id: parseInt(threadId, 10) }
        : {};

      // Telegram has a 4096 character limit per message — split if needed
      const MAX_LENGTH = 4096;
      if (text.length <= MAX_LENGTH) {
        await sendTelegramMessage(this.bot.api, numericId, text, options);
      } else {
        for (let i = 0; i < text.length; i += MAX_LENGTH) {
          await sendTelegramMessage(
            this.bot.api,
            numericId,
            text.slice(i, i + MAX_LENGTH),
            options,
          );
        }
      }
      logger.info(
        { jid, length: text.length, threadId },
        'Telegram message sent',
      );
    } catch (err) {
      logger.error({ jid, err }, 'Failed to send Telegram message');
    }
  }

  isConnected(): boolean {
    return this.bot !== null;
  }

  ownsJid(jid: string): boolean {
    return jid.startsWith('tg:');
  }

  async disconnect(): Promise<void> {
    if (this.bot) {
      this.bot.stop();
      this.bot = null;
      logger.info('Telegram bot stopped');
    }
  }

  async setTyping(jid: string, isTyping: boolean): Promise<void> {
    if (!this.bot || !isTyping) return;
    try {
      const numericId = jid.replace(/^tg:/, '');
      await this.bot.api.sendChatAction(numericId, 'typing');
    } catch (err) {
      logger.debug({ jid, err }, 'Failed to send Telegram typing indicator');
    }
  }
}

registerChannel('telegram', (opts: ChannelOpts) => {
  const envVars = readEnvFile(['TELEGRAM_BOT_TOKEN']);
  const token =
    process.env.TELEGRAM_BOT_TOKEN || envVars.TELEGRAM_BOT_TOKEN || '';
  if (!token) {
    logger.warn('Telegram: TELEGRAM_BOT_TOKEN not set');
    return null;
  }
  return new TelegramChannel(token, opts);
});
