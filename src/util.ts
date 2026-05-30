import { randomBytes } from 'node:crypto';
import {
  ChatCompletionChunk,
  ChatCompletionMessageParam,
  ChatCompletionTool,
} from 'openai/resources';
import { z } from 'zod/v4';

const RaycastRequestTool = z.discriminatedUnion('type', [
  z.object({
    name: z.string(),
    type: z.literal('remote_tool'),
  }),
  z.object({
    type: z.literal('local_tool'),
    function: z.object({
      name: z.string(),
      description: z.string(),
      parameters: z.record(z.string(), z.any()).transform((value) => {
        if (Object.keys(value).length === 0) {
          return {
            type: 'object',
            properties: {},
            required: [],
          };
        }
        return value;
      }),
    }),
  }),
]);
type RaycastRequestTool = z.infer<typeof RaycastRequestTool>;

export const OllamaChatMessage = z.object({
  role: z.enum(['user', 'assistant', 'system', 'tool']),
  images: z.array(z.string()).optional(),
  content: z.string(),
  tool_calls: z
    .array(
      z.record(
        z.literal('function'),
        z.object({
          name: z.string(),
          arguments: z.record(z.string(), z.any()),
        }),
      ),
    )
    .optional(),
});
type OllamaChatMessage = z.infer<typeof OllamaChatMessage>;

export const OllamaChatRequest = z.object({
  model: z.string(),
  messages: z.array(OllamaChatMessage),
  tools: z.array(RaycastRequestTool).default([]),
});

export interface OllamaChunkResponse {
  model: string;
  created_at: string;
  message: {
    role: 'assistant';
    content: string;
    thinking?: string;
    tool_calls?: {
      function: {
        name: string;
        arguments: Record<string, unknown>;
      };
    }[];
  };
  done: boolean;
  done_reason?: 'stop' | 'tool_calls';
}

export function makeOllamaChunk(
  model: string,
  data: { content: string; thinking?: string },
  done: boolean,
  done_reason?: OllamaChunkResponse['done_reason'],
  toolCalls?: Record<number, ChatCompletionChunk.Choice.Delta.ToolCall>,
): OllamaChunkResponse {
  // Convert tools to Ollama format
  const finalToolCalls: OllamaChunkResponse['message']['tool_calls'] = [];
  if (toolCalls) {
    for (const key in toolCalls) {
      const tc = toolCalls[key];
      if (!tc.function?.name) {
        continue;
      }

      const args = tc.function.arguments || '{}';
      try {
        const parsedArgs = JSON.parse(args);
        finalToolCalls.push({
          function: {
            name: tc.function.name,
            arguments: parsedArgs,
          },
        });
      } catch {
        continue;
      }
    }
  }

  return {
    model,
    created_at: new Date().toISOString(),
    message: {
      role: 'assistant',
      ...data,
      tool_calls: finalToolCalls.length > 0 ? finalToolCalls : undefined,
    },
    done,
    done_reason,
  };
}

export function convertOllamaMessagesToOpenAI(
  messages: OllamaChatMessage[],
): ChatCompletionMessageParam[] {
  // Store all tool call IDs in order
  const toolCallIds: string[] = [];

  const makeToolCallId = (): string => {
    return randomBytes(5).toString('hex').slice(0, 9);
  };

  return messages.map((msg): ChatCompletionMessageParam => {
    // Handle tool calls in assistant messages
    if (msg.role === 'assistant' && msg.tool_calls) {
      // Clear previous tool call IDs and generate new ones
      toolCallIds.length = 0;

      return {
        role: 'assistant',
        content: msg.content,
        // @ts-ignore - Bypass for DeepSeek's strict thinking mode validation
        reasoning_content: '',
        tool_calls: msg.tool_calls.map((tc) => {
          const toolCallId = makeToolCallId();
          toolCallIds.push(toolCallId); // Store each tool call ID
          return {
            id: toolCallId,
            type: 'function',
            function: {
              name: tc.function.name,
              arguments: JSON.stringify(tc.function.arguments),
            },
          };
        }),
      };
    }

    // Handle tool responses
    if (msg.role === 'tool') {
      // Use the next available tool call ID in sequence
      const toolCallId = toolCallIds.shift() || makeToolCallId();
      return {
        role: 'tool',
        content: msg.content,
        tool_call_id: toolCallId,
      };
    }

    // Handle images if present
    if (msg.images && msg.images.length > 0 && msg.role === 'user') {
      return {
        role: 'user',
        content: [
          { type: 'text', text: msg.content },
          ...msg.images.map((img) => ({
            type: 'image_url' as const,
            image_url: { url: `data:image/jpeg;base64,${img}` },
          })),
        ],
      };
    }

    // Handle regular assistant messages (without tool calls)
    if (msg.role === 'assistant') {
      return {
        role: 'assistant',
        content: msg.content,
        // @ts-ignore - Bypass for DeepSeek's strict thinking mode validation
        reasoning_content: '',
      } as ChatCompletionMessageParam;
    }

    // Handle regular messages (user, system)
    return {
      role: msg.role as 'user' | 'system',
      content: msg.content,
    } as ChatCompletionMessageParam;
  });
}

export function convertRaycastToolsToOpenAI(
  raycastTools?: RaycastRequestTool[],
): ChatCompletionTool[] | undefined {
  const filteredTools = raycastTools?.filter((tool) => tool.type === 'local_tool');

  if (!filteredTools || filteredTools.length === 0) {
    return undefined;
  }

  return filteredTools.map((tool) => {
    return {
      type: 'function',
      function: tool.function,
    };
  });
}

export function makeSSEMessage(message: OllamaChunkResponse): string {
  return `${JSON.stringify(message)}\n\n`;
}

type ThinkingDelta =
  | (ChatCompletionChunk.Choice.Delta & {
      reasoning?: unknown;
      reasoning_content?: unknown;
      extra_content?: {
        google?: {
          thought?: unknown;
        };
      };
    })
  | undefined;

export function getThoughtsFromResponseDelta(delta: ThinkingDelta): string | null {
  if (!delta) {
    return null;
  }

  // Some providers use reasoning or reasoning_content
  const reasoning = delta.reasoning ?? delta.reasoning_content;
  if (typeof reasoning === 'string' && reasoning.length > 0) {
    return reasoning;
  }

  // Gemini API
  const googleThought = delta.extra_content?.google?.thought;
  if (typeof googleThought === 'boolean' && googleThought) {
    const content = delta.content ?? '';
    const thinking = content.replace(/<thought>|<\/thought>/g, '');
    if (thinking.length > 0) {
      return thinking;
    }
  }

  return null;
}
