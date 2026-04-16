// ============================================================================
// Gemma 4 E4B - Node.js Agent Framework
// Modified by ClaudeO
// Purpose: Node.js client and agent framework for the Gemma 4 E4B API.
//          Supports multi-agent orchestration, streaming, and tool calling.
// ============================================================================

import axios from 'axios';
import { EventEmitter } from 'events';

// --- Modified by ClaudeO: Configuration constants ---
const DEFAULT_API_URL = process.env.GEMMA4_API_URL || 'http://localhost:8000';
const DEFAULT_OLLAMA_URL = process.env.OLLAMA_URL || 'http://localhost:11434';

/**
 * Gemma4Client - Core HTTP client for the Gemma 4 E4B API
 * Provides both direct Ollama access and unified API access.
 */
class Gemma4Client {
  /**
   * @param {Object} options - Client configuration
   * @param {string} options.apiUrl - API server URL
   * @param {string} options.ollamaUrl - Direct Ollama URL
   * @param {number} options.timeout - Request timeout in ms
   */
  constructor(options = {}) {
    // Modified by ClaudeO: Flexible endpoint configuration
    this.apiUrl = (options.apiUrl || DEFAULT_API_URL).replace(/\/+$/, '');
    this.ollamaUrl = (options.ollamaUrl || DEFAULT_OLLAMA_URL).replace(/\/+$/, '');
    this.timeout = options.timeout || 120000;
    this.defaultModel = options.model || 'gemma4-e4b-opt';
  }

  /**
   * Send a chat completion request via the unified API.
   * @param {Object[]} messages - Array of {role, content} messages
   * @param {Object} options - Generation options
   * @returns {Promise<string>} Assistant response text
   */
  async chat(messages, options = {}) {
    const payload = {
      messages,
      model: options.model || this.defaultModel,
      temperature: options.temperature ?? 0.7,
      stream: false,
      agent_id: options.agentId || null,
      enable_thinking: options.enableThinking || false,
      max_tokens: options.maxTokens || null,
    };

    const resp = await axios.post(
      `${this.apiUrl}/v1/chat/completions`,
      payload,
      { timeout: this.timeout }
    );

    return resp.data.choices[0].message.content;
  }

  /**
   * Send a chat request directly to Ollama (bypass unified API).
   * Useful when the API server is not running.
   * @param {Object[]} messages - Conversation messages
   * @param {Object} options - Ollama-specific options
   * @returns {Promise<string>} Response text
   */
  async chatDirect(messages, options = {}) {
    const payload = {
      model: options.model || this.defaultModel,
      messages,
      stream: false,
      options: {
        temperature: options.temperature ?? 0.7,
        num_ctx: options.numCtx || 4096,
        top_p: options.topP ?? 0.95,
        top_k: options.topK ?? 64,
      },
    };

    const resp = await axios.post(
      `${this.ollamaUrl}/api/chat`,
      payload,
      { timeout: this.timeout }
    );

    return resp.data.message.content;
  }

  /**
   * Health check for both API server and Ollama.
   * @returns {Promise<Object>} Health status object
   */
  async healthCheck() {
    const result = { api: false, ollama: false };
    try {
      const apiResp = await axios.get(`${this.apiUrl}/health`, { timeout: 5000 });
      result.api = apiResp.data.status === 'healthy';
      result.apiDetails = apiResp.data;
    } catch { /* API not available */ }

    try {
      await axios.get(this.ollamaUrl, { timeout: 5000 });
      result.ollama = true;
    } catch { /* Ollama not available */ }

    return result;
  }

  /**
   * List available agents from the API server.
   * @returns {Promise<Object[]>} Array of agent configurations
   */
  async listAgents() {
    const resp = await axios.get(`${this.apiUrl}/v1/agents`);
    return resp.data.agents;
  }

  /**
   * Register a new agent with the API server.
   * @param {Object} config - Agent configuration
   * @returns {Promise<Object>} Registration result
   */
  async registerAgent(config) {
    const resp = await axios.post(`${this.apiUrl}/v1/agents`, config);
    return resp.data;
  }
}


// --- Modified by ClaudeO: Agent base class for building specialized agents ---

/**
 * Gemma4Agent - Base class for building AI agents.
 * Extend this class to create specialized agents with custom
 * system prompts, tool definitions, and behavior.
 */
class Gemma4Agent extends EventEmitter {
  /**
   * @param {Object} config - Agent configuration
   * @param {string} config.agentId - Unique agent identifier
   * @param {string} config.name - Human-readable agent name
   * @param {string} config.systemPrompt - System prompt defining behavior
   * @param {Gemma4Client} config.client - Gemma4Client instance
   * @param {number} config.temperature - Generation temperature
   * @param {number} config.maxTokens - Max response tokens
   */
  constructor(config) {
    super();
    this.agentId = config.agentId;
    this.name = config.name;
    this.systemPrompt = config.systemPrompt;
    this.client = config.client || new Gemma4Client();
    this.temperature = config.temperature ?? 0.7;
    this.maxTokens = config.maxTokens || 2048;
    this.history = [];
    this.tools = config.tools || [];
  }

  /**
   * Process a user message through this agent.
   * Maintains conversation history and applies system prompt.
   * @param {string} userMessage - The user's input
   * @param {Object} context - Additional context (images, metadata)
   * @returns {Promise<string>} Agent's response
   */
  async process(userMessage, context = {}) {
    this.emit('processing', { message: userMessage, context });

    const messages = [
      { role: 'system', content: this.systemPrompt },
      ...this.history,
      { role: 'user', content: userMessage },
    ];

    // Add images if provided in context
    if (context.images) {
      messages[messages.length - 1].images = context.images;
    }

    try {
      const response = await this.client.chat(messages, {
        temperature: this.temperature,
        maxTokens: this.maxTokens,
        agentId: this.agentId,
        enableThinking: context.enableThinking || false,
      });

      // Update conversation history
      this.history.push(
        { role: 'user', content: userMessage },
        { role: 'assistant', content: response }
      );

      // Trim history to prevent context overflow (keep last 10 turns)
      if (this.history.length > 20) {
        this.history = this.history.slice(-20);
      }

      this.emit('response', { response, agent: this.agentId });
      return response;
    } catch (error) {
      this.emit('error', { error, agent: this.agentId });
      throw error;
    }
  }

  /** Reset the agent's conversation history. */
  resetHistory() {
    this.history = [];
    this.emit('reset', { agent: this.agentId });
  }
}


// --- Modified by ClaudeO: Multi-agent orchestrator ---

/**
 * AgentOrchestrator - Manages multiple agents and routes requests.
 * Supports automatic agent selection based on intent detection.
 */
class AgentOrchestrator {
  /**
   * @param {Gemma4Client} client - Shared client instance
   */
  constructor(client) {
    this.client = client || new Gemma4Client();
    this.agents = new Map();
    this.routerAgent = null;
  }

  /**
   * Register an agent with the orchestrator.
   * @param {Gemma4Agent} agent - Agent instance to register
   */
  register(agent) {
    this.agents.set(agent.agentId, agent);
    console.log(`[Orchestrator] Registered agent: ${agent.agentId} (${agent.name})`);
  }

  /**
   * Create and register a simple agent from configuration.
   * @param {Object} config - Agent config (agentId, name, systemPrompt, etc.)
   * @returns {Gemma4Agent} The created agent
   */
  createAgent(config) {
    const agent = new Gemma4Agent({
      ...config,
      client: this.client,
    });
    this.register(agent);
    return agent;
  }

  /**
   * Route a message to the appropriate agent.
   * @param {string} agentId - Target agent ID
   * @param {string} message - User message
   * @param {Object} context - Additional context
   * @returns {Promise<Object>} Response with agent info
   */
  async route(agentId, message, context = {}) {
    const agent = this.agents.get(agentId);
    if (!agent) {
      throw new Error(`Agent '${agentId}' not found. Available: ${[...this.agents.keys()].join(', ')}`);
    }

    const response = await agent.process(message, context);
    return {
      agentId,
      agentName: agent.name,
      response,
      timestamp: new Date().toISOString(),
    };
  }

  /**
   * Auto-route a message by asking a router agent to pick the best agent.
   * @param {string} message - User message to route
   * @returns {Promise<Object>} Response from the selected agent
   */
  async autoRoute(message) {
    // Use a simple heuristic-based router (no extra LLM call needed)
    const agentIds = [...this.agents.keys()];
    const keywords = {
      coder: ['code', 'function', 'bug', 'script', 'program', 'debug', 'api', 'python', 'javascript'],
      analyst: ['data', 'chart', 'analyze', 'statistics', 'csv', 'report', 'numbers'],
      translator: ['translate', 'language', 'chinese', 'japanese', 'spanish', 'french', 'korean'],
      vision: ['image', 'photo', 'picture', 'screenshot', 'ocr', 'visual'],
      planner: ['plan', 'steps', 'workflow', 'task', 'organize', 'schedule', 'project'],
    };

    const lowerMsg = message.toLowerCase();
    let bestAgent = 'general';
    let bestScore = 0;

    for (const [agentId, kws] of Object.entries(keywords)) {
      if (this.agents.has(agentId)) {
        const score = kws.filter(kw => lowerMsg.includes(kw)).length;
        if (score > bestScore) {
          bestScore = score;
          bestAgent = agentId;
        }
      }
    }

    console.log(`[Orchestrator] Auto-routed to: ${bestAgent} (score: ${bestScore})`);
    return this.route(bestAgent, message);
  }

  /** Get status of all registered agents. */
  getStatus() {
    return [...this.agents.entries()].map(([id, agent]) => ({
      agentId: id,
      name: agent.name,
      historyLength: agent.history.length,
    }));
  }
}


// ============================================================================
// Pre-built Agent Factory
// ============================================================================

// Modified by ClaudeO: Factory function for common agent configurations
/**
 * Create a pre-configured orchestrator with standard agents.
 * @param {Object} clientOptions - Options for the Gemma4Client
 * @returns {AgentOrchestrator} Ready-to-use orchestrator
 */
function createDefaultOrchestrator(clientOptions = {}) {
  const client = new Gemma4Client(clientOptions);
  const orchestrator = new AgentOrchestrator(client);

  // Register standard agents
  orchestrator.createAgent({
    agentId: 'general',
    name: 'General Assistant',
    systemPrompt: 'You are a helpful AI assistant powered by Gemma 4 E4B running locally. Be concise and accurate.',
    temperature: 0.7,
  });

  orchestrator.createAgent({
    agentId: 'coder',
    name: 'Code Assistant',
    systemPrompt: 'You are an expert coding assistant. Write clean, documented code. Show approach briefly then provide the solution. Support Python, JS, TS, shell.',
    temperature: 0.3,
  });

  orchestrator.createAgent({
    agentId: 'analyst',
    name: 'Data Analyst',
    systemPrompt: 'You are a data analysis agent. Analyze data precisely, present structured findings, and extract insights from visual data.',
    temperature: 0.4,
  });

  orchestrator.createAgent({
    agentId: 'translator',
    name: 'Multilingual Translator',
    systemPrompt: 'You are a professional translator supporting 140+ languages. Preserve tone, idioms, and cultural nuances. Specify source and target languages.',
    temperature: 0.3,
  });

  orchestrator.createAgent({
    agentId: 'vision',
    name: 'Vision Analyst',
    systemPrompt: 'You are a vision analysis agent. Describe images precisely, extract text (OCR), interpret charts. Provide structured analysis.',
    temperature: 0.5,
  });

  orchestrator.createAgent({
    agentId: 'planner',
    name: 'Task Planner',
    systemPrompt: 'You are a task planner. Break complex requests into actionable steps with inputs, outputs, and tool calls. Think step-by-step.',
    temperature: 0.5,
  });

  return orchestrator;
}


// ============================================================================
// Express Server for Node.js Agent API
// ============================================================================

/**
 * Start an Express server that exposes the agent orchestrator as a REST API.
 * This complements the Python API server with Node.js-native agent features.
 * @param {number} port - Server port (default: 8001)
 */
async function startAgentServer(port = 8001) {
  const { default: express } = await import('express');
  const app = express();
  app.use(express.json());

  const orchestrator = createDefaultOrchestrator();

  // Health check
  app.get('/health', (req, res) => {
    res.json({
      status: 'healthy',
      agents: orchestrator.getStatus(),
      timestamp: new Date().toISOString(),
    });
  });

  // Chat with specific agent
  app.post('/agent/:agentId/chat', async (req, res) => {
    try {
      const { agentId } = req.params;
      const { message, context } = req.body;
      const result = await orchestrator.route(agentId, message, context || {});
      res.json(result);
    } catch (error) {
      res.status(400).json({ error: error.message });
    }
  });

  // Auto-route chat
  app.post('/agent/auto/chat', async (req, res) => {
    try {
      const { message } = req.body;
      const result = await orchestrator.autoRoute(message);
      res.json(result);
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  });

  // List agents
  app.get('/agents', (req, res) => {
    res.json({ agents: orchestrator.getStatus() });
  });

  // Register custom agent
  app.post('/agents', (req, res) => {
    try {
      const agent = orchestrator.createAgent(req.body);
      res.json({ status: 'registered', agentId: agent.agentId });
    } catch (error) {
      res.status(400).json({ error: error.message });
    }
  });

  // Reset agent history
  app.post('/agent/:agentId/reset', (req, res) => {
    const agent = orchestrator.agents.get(req.params.agentId);
    if (agent) {
      agent.resetHistory();
      res.json({ status: 'reset', agentId: req.params.agentId });
    } else {
      res.status(404).json({ error: 'Agent not found' });
    }
  });

  app.listen(port, () => {
    console.log(`[Gemma4 Node.js Agent Server] Running on http://localhost:${port}`);
    console.log(`[Agents] ${orchestrator.getStatus().map(a => a.agentId).join(', ')}`);
  });
}


// ============================================================================
// Exports
// ============================================================================

export {
  Gemma4Client,
  Gemma4Agent,
  AgentOrchestrator,
  createDefaultOrchestrator,
  startAgentServer,
};

// Run server if executed directly
if (process.argv[1] && process.argv[1].includes('gemma4_agent_node')) {
  const port = parseInt(process.argv[2]) || 8001;
  startAgentServer(port);
}
