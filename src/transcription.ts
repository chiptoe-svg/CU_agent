import { execFile } from 'child_process';
import fs from 'fs';
import path from 'path';
import { promisify } from 'util';

import { logger } from './logger.js';

const execFileAsync = promisify(execFile);

const WHISPER_BIN = process.env.WHISPER_BIN || 'whisper-cli';
const WHISPER_MODEL =
  process.env.WHISPER_MODEL ||
  path.resolve(process.cwd(), 'data/models/ggml-base.bin');

/**
 * Transcribe an audio file using local whisper.cpp.
 * Converts to 16kHz mono WAV first via ffmpeg, then runs whisper-cli.
 * Returns the transcript text, or null on failure.
 */
export async function transcribeAudio(
  audioPath: string,
): Promise<string | null> {
  const wavPath = audioPath.replace(/\.[^.]+$/, '.wav');

  try {
    // Convert to 16kHz mono WAV (whisper.cpp requirement)
    await execFileAsync(
      'ffmpeg',
      ['-i', audioPath, '-ar', '16000', '-ac', '1', '-f', 'wav', '-y', wavPath],
      { timeout: 30_000 },
    );

    // Run whisper-cli
    const { stdout } = await execFileAsync(
      WHISPER_BIN,
      ['-m', WHISPER_MODEL, '-f', wavPath, '--no-timestamps', '-nt'],
      { timeout: 60_000 },
    );

    const transcript = stdout.trim();
    if (!transcript) {
      logger.warn({ audioPath }, 'Whisper returned empty transcript');
      return null;
    }

    logger.info(
      { audioPath, length: transcript.length },
      'Transcribed voice message',
    );
    return transcript;
  } catch (err) {
    logger.error({ audioPath, err }, 'whisper.cpp transcription failed');
    return null;
  } finally {
    // Clean up the intermediate WAV file
    try {
      if (fs.existsSync(wavPath)) fs.unlinkSync(wavPath);
    } catch {
      /* ignore */
    }
  }
}
