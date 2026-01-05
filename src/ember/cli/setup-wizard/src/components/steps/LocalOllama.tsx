import React, {useEffect, useState} from 'react';
import {Box, Text, useInput} from 'ink';
import TextInput from 'ink-text-input';
import SelectInput from 'ink-select-input';
import {saveConfig} from '../../utils/config.js';

interface Props {
  onBack?: () => void;
  onSkip?: () => void;
  onCancel?: () => void;
  onComplete?: (exampleFile?: string) => void;
}

type Reachability = 'unknown' | 'ok' | 'error';

export const LocalOllama: React.FC<Props> = ({onBack, onSkip, onCancel, onComplete}) => {
  const [baseUrl, setBaseUrl] = useState<string>(process.env.OLLAMA_BASE_URL || 'http://localhost:11434');
  const [editingUrl, setEditingUrl] = useState<boolean>(false);
  const [reach, setReach] = useState<Reachability>('unknown');
  const [reachMsg, setReachMsg] = useState<string>('Checking...');
  const [model, setModel] = useState<string>('llama3.2:1b');
  const [pulling, setPulling] = useState<boolean>(false);
  const [pullMsg, setPullMsg] = useState<string>('');
  const [done, setDone] = useState<boolean>(false);

  const models = [
    { label: 'llama3.2:1b (fast, good default)', value: 'llama3.2:1b' },
    { label: 'qwen2.5:0.5b (smallest, very fast)', value: 'qwen2.5:0.5b' },
    { label: 'tinyllama (tiny, weakest quality)', value: 'tinyllama' },
  ];

  async function checkReachability(url: string) {
    try {
      setReach('unknown');
      setReachMsg('Checking...');
      const res = await fetch(`${url}/api/version`, { method: 'GET' });
      if (res.ok) {
        setReach('ok');
        setReachMsg('Ollama is reachable.');
      } else {
        setReach('error');
        setReachMsg(`HTTP ${res.status}`);
      }
    } catch (error: unknown) {
      setReach('error');
      const message = error instanceof Error ? error.message : undefined;
      if (!message || /fetch failed/i.test(message) || /failed to fetch/i.test(message)) {
        const fallbackMsg =
          `Could not reach Ollama at ${url}. Install and start the Ollama runtime ` +
          '(run "ollama serve") or update the base URL.';
        setReachMsg(fallbackMsg);
      } else {
        setReachMsg(message);
      }
    }
  }

  useEffect(() => {
    checkReachability(baseUrl);
  }, []);

  useInput((input, key) => {
    if (key.escape) onCancel?.();
    if ((input === 'b' || input === 'B') && onBack) onBack();
    if ((input === 's' || input === 'S') && onSkip) onSkip();
    if (input === 'e' || input === 'E') setEditingUrl(true);
    if (input === 'r' || input === 'R') checkReachability(baseUrl);
    if ((input === 'c' || input === 'C') && reach === 'ok' && !pulling) startPull();
  });

  async function startPull() {
    setPulling(true);
    setPullMsg('Starting model pull...');
    try {
      // stream /api/pull
      const body = { model, stream: true };
      const res = await fetch(`${baseUrl}/api/pull`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      if (!res.body) throw new Error('No response body');
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let ok = false;
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const text = decoder.decode(value, { stream: true });
        for (const line of text.split('\n')) {
          if (!line.trim()) continue;
          try {
            const ev = JSON.parse(line);
            const status = String(ev.status || '').toLowerCase();
            if (typeof ev.completed === 'number' && typeof ev.total === 'number' && ev.total > 0) {
              const pct = Math.min(100, Math.round((ev.completed / ev.total) * 100));
              setPullMsg(`Pulling ${model}... ${pct}%`);
            } else if (status) {
              setPullMsg(status);
            }
            if (status === 'success' || status === 'ok' || status === 'done' || (ev.completed && ev.total && ev.completed >= ev.total)) {
              ok = true;
            }
          } catch {
            // ignore line parse errors
          }
        }
      }
      if (!ok) throw new Error('Pull did not complete successfully');

      // Save config: default provider/model and base URL
      await saveConfig({
        default_provider: 'ollama',
        providers: {
          ollama: {
            default_model: model,
            base_url: baseUrl,
          }
        }
      });

      setPullMsg('Done. Saved configuration.');
      setDone(true);
      setTimeout(() => onComplete?.('hello_ember.py'), 500);
    } catch (e: any) {
      setPullMsg(`Error: ${e?.message || e}`);
      setPulling(false);
    }
  }

  return (
    <Box flexDirection="column">
      <Box marginBottom={1}>
        <Text bold color="magenta">Use local Ollama (no API keys)</Text>
      </Box>

      <Box marginBottom={1}>
        <Text>Base URL: </Text>
        {editingUrl ? (
          <TextInput value={baseUrl} onChange={setBaseUrl} onSubmit={(v) => { setBaseUrl(v); setEditingUrl(false); checkReachability(v); }} />
        ) : (
          <Text color="cyan">{baseUrl}</Text>
        )}
      </Box>

      <Box marginBottom={1}>
        <Text dimColor>{reach === 'ok' ? '✓' : reach === 'error' ? '✗' : '…'} {reachMsg}</Text>
      </Box>

      <Box marginTop={1} flexDirection="column">
        <Text>Select a starter model:</Text>
        <Box marginTop={1}>
          <SelectInput
            items={models}
            onSelect={(item) => setModel(item.value)}
            isFocused={!editingUrl && !pulling}
          />
        </Box>
      </Box>

      <Box marginTop={1}>
        {!done && pulling && <Text color="yellow">{pullMsg}</Text>}
        {!pulling && <Text dimColor>Press C to continue and pull the selected model</Text>}
      </Box>

      <Box marginTop={2} flexDirection="column">
        <Box>
          <Text dimColor>E</Text><Text dimColor> - Edit base URL</Text>
        </Box>
        <Box>
          <Text dimColor>R</Text><Text dimColor> - Retry reachability</Text>
        </Box>
        <Box>
          <Text dimColor>C</Text><Text dimColor> - Continue (pull + save config)</Text>
        </Box>
        {onBack && (
          <Box>
            <Text dimColor>B</Text><Text dimColor> - Back</Text>
          </Box>
        )}
        {onSkip && (
          <Box>
            <Text dimColor>S</Text><Text dimColor> - Skip Ollama setup</Text>
          </Box>
        )}
        <Box>
          <Text dimColor>ESC</Text><Text dimColor> - Cancel</Text>
        </Box>
      </Box>
    </Box>
  );
};
