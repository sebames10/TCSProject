
import json
import ssl
import urllib.request
import os

# ---------------------------------------------------------------------------
# Fake OpenAI client that routes to the LiteLLM proxy server.
# Uses only Python stdlib — no extra packages needed.
#
# Supports:
#   client.responses.create(...)          — Responses API (output_text)
#   client.chat.completions.create(...)   — Chat Completions API
#   client.embeddings.create(...)         — Embeddings API
# ---------------------------------------------------------------------------

MODEL_MAP = {
    "gpt-4.1-mini": "azure/genailab-maas-gpt-4.1-mini",
    "gpt-4o":       "genailab-maas-gpt-4o",
    "gpt-4.1":      "azure/genailab-maas-gpt-4.1",
    "gpt-4.1-nano": "azure/genailab-maas-gpt-4.1-nano",
    "gpt-4o-mini":  "azure/genailab-maas-gpt-4o-mini",
    "text-embedding-3-small": "azure/genailab-maas-text-embedding-3-large",
    "text-embedding-3-large": "azure/genailab-maas-text-embedding-3-large",
}

_SSL_CTX = ssl._create_unverified_context()


def _post(url, payload, api_key):
    """Send a JSON POST request and return the parsed response body."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "openai-python/1.50.0"
        },
    )
    with urllib.request.urlopen(req, context=_SSL_CTX, timeout=60) as r:
        return json.loads(r.read())


# ── Responses API ────────────────────────────────────────────────────────────

class _FakeResponse:
    """Mimics the OpenAI Responses API response object (.output_text)."""
    def __init__(self, text):
        self.output_text = text


class _FakeResponsesNamespace:
    def __init__(self, api_key, base_url):
        self._api_key  = api_key
        self._base_url = base_url.rstrip("/")

    def create(self, model, input, temperature=0.7, top_p=1.0,
               max_output_tokens=120, seed=None, **kwargs):
        resolved_model = MODEL_MAP.get(model, model)
        payload = {
            "model":       resolved_model,
            "messages":    input,
            "temperature": temperature,
            "top_p":       top_p,
            "max_tokens":  max_output_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        body = _post(f"{self._base_url}/chat/completions", payload, self._api_key)
        return _FakeResponse(body["choices"][0]["message"]["content"])

    def create_embedding(self, input):
        """Embed a single string and return an object with .data[0].embedding."""
        resolved_model = MODEL_MAP.get("text-embedding-3-small")
        payload = {
            "model": resolved_model,
            "input": input if isinstance(input, list) else [input],
        }
        body = _post(f"{self._base_url}/embeddings", payload, self._api_key)
        return _EmbeddingResponse(body)


# ── Chat Completions API ─────────────────────────────────────────────────────

class _ToolCallFunction:
    """Mimics openai.types.chat.chat_completion_message_tool_call.Function."""
    def __init__(self, name, arguments):
        self.name      = name
        self.arguments = arguments  # raw JSON string, as the real SDK returns


class _ToolCall:
    """Mimics openai.types.chat.ChatCompletionMessageToolCall."""
    def __init__(self, tool_call_dict):
        self.id       = tool_call_dict.get("id", "")
        self.type     = tool_call_dict.get("type", "function")
        self.function = _ToolCallFunction(
            name      = tool_call_dict["function"]["name"],
            arguments = tool_call_dict["function"]["arguments"],
        )

    def model_dump(self):
        return {
            "id":       self.id,
            "type":     self.type,
            "function": {
                "name":      self.function.name,
                "arguments": self.function.arguments,
            },
        }


class _ChatMessage:
    """Mimics openai.types.chat.ChatCompletionMessage."""
    def __init__(self, msg_dict):
        self.role       = msg_dict.get("role", "assistant")
        self.content    = msg_dict.get("content")  # None when tool_calls present
        raw_tool_calls  = msg_dict.get("tool_calls") or []
        self.tool_calls = [_ToolCall(tc) for tc in raw_tool_calls] or None

    def model_dump(self):
        """Allow appending this message back to the conversation as a plain dict."""
        d = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [tc.model_dump() for tc in self.tool_calls]
        return d

    # Make it directly appendable to a messages list (dict-like)
    def __getitem__(self, key):
        return self.model_dump()[key]

    def keys(self):
        return self.model_dump().keys()


class _ChatChoice:
    def __init__(self, choice_dict):
        self.message       = _ChatMessage(choice_dict["message"])
        self.finish_reason = choice_dict.get("finish_reason")


class _ChatResponse:
    """Mimics openai.types.chat.ChatCompletion — supports tool_calls."""
    def __init__(self, body):
        self.choices = [_ChatChoice(c) for c in body["choices"]]


class _FakeCompletionsNamespace:
    def __init__(self, api_key, base_url):
        self._api_key  = api_key
        self._base_url = base_url.rstrip("/")

    def create(self, model, messages, temperature=0.7, top_p=1.0,
               max_tokens=None, seed=None, tools=None, tool_choice=None, **kwargs):
        resolved_model = MODEL_MAP.get(model, model)

        # Serialize any _ChatMessage objects back to plain dicts
        serialized_messages = []
        for m in messages:
            if isinstance(m, _ChatMessage):
                serialized_messages.append(m.model_dump())
            elif hasattr(m, "model_dump"):
                serialized_messages.append(m.model_dump())
            else:
                serialized_messages.append(m)

        payload = {
            "model":    resolved_model,
            "messages": serialized_messages,
            "temperature": temperature,
            "top_p":       top_p,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if seed is not None:
            payload["seed"] = seed
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        body = _post(f"{self._base_url}/chat/completions", payload, self._api_key)
        return _ChatResponse(body)


class _FakeChatNamespace:
    def __init__(self, api_key, base_url):
        self.completions = _FakeCompletionsNamespace(api_key, base_url)


# ── Embeddings API ───────────────────────────────────────────────────────────

class _EmbeddingObject:
    def __init__(self, embedding):
        self.embedding = embedding


class _EmbeddingResponse:
    """Mimics openai.types.CreateEmbeddingResponse (.data[0].embedding)."""
    def __init__(self, body):
        self.data = [_EmbeddingObject(item["embedding"]) for item in body["data"]]


class _FakeEmbeddingsNamespace:
    def __init__(self, api_key, base_url):
        self._api_key  = api_key
        self._base_url = base_url.rstrip("/")

    def create(self, model, input, **kwargs):
        resolved_model = MODEL_MAP.get(model, model)
        payload = {
            "model": resolved_model,
            "input": input if isinstance(input, list) else [input],
        }
        body = _post(f"{self._base_url}/embeddings", payload, self._api_key)
        return _EmbeddingResponse(body)


# ── Public OpenAI class ──────────────────────────────────────────────────────

class OpenAI:
    """Drop-in replacement for openai.OpenAI that points to the LiteLLM server."""
    def __init__(self, api_key=None, base_url=None, **kwargs):
        _key  = api_key  or os.environ.get("LITELLM_API_KEY", "")
        _base = base_url or os.environ.get("LITELLM_BASE_URL", "https://genailab.tcs.in")
        self.responses  = _FakeResponsesNamespace(_key, _base)
        self.chat       = _FakeChatNamespace(_key, _base)
        self.embeddings = _FakeEmbeddingsNamespace(_key, _base)


# ── LangChain-compatible ChatOpenAI replacement ──────────────────────────────

def _serialize_lc_messages(messages):
    """Convert LangChain message objects (or plain dicts/strings) to a list of dicts."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    serialized = []
    for msg in messages:
        if isinstance(msg, dict):
            serialized.append(msg)
            continue
        msg_type = getattr(msg, "type", None)
        if msg_type == "system":
            role = "system"
        elif msg_type == "human":
            role = "user"
        elif msg_type in ("ai", "assistant"):
            role = "assistant"
        elif msg_type == "tool":
            # LangChain ToolMessage
            serialized.append({
                "role":         "tool",
                "content":      str(msg.content),
                "tool_call_id": getattr(msg, "tool_call_id", ""),
            })
            continue
        else:
            cls = type(msg).__name__.lower()
            role = "system" if "system" in cls else "assistant" if "ai" in cls else "user"

        # AIMessage may carry tool_calls
        raw_tc = getattr(msg, "tool_calls", None)
        if role == "assistant" and raw_tc:
            # LangChain format: [{"name", "args", "id"}]
            tc_list = []
            for tc in raw_tc:
                tc_list.append({
                    "id":   tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name":      tc["name"],
                        "arguments": json.dumps(tc.get("args", {})),
                    },
                })
            serialized.append({"role": "assistant", "content": None, "tool_calls": tc_list})
        else:
            serialized.append({"role": role, "content": getattr(msg, "content", str(msg))})
    return serialized


def _lc_tool_schema(lc_tool):
    """Convert a LangChain @tool object into an OpenAI tool schema dict."""
    schema = lc_tool.args_schema.model_json_schema() if hasattr(lc_tool, "args_schema") else {}
    schema.pop("title", None)
    return {
        "type": "function",
        "function": {
            "name":        lc_tool.name,
            "description": lc_tool.description,
            "parameters":  schema,
        },
    }


try:
    from langchain_core.messages import AIMessage as _LCBaseAIMessage
    _LC_AIMESSAGE_BASE = _LCBaseAIMessage
except ImportError:
    _LC_AIMESSAGE_BASE = object


class _LangChainAIMessage(_LC_AIMESSAGE_BASE):
    """
    Mimics langchain_core.messages.AIMessage.
    Inherits from the real AIMessage (a BaseMessage) when langchain_core is available
    so that StrOutputParser and other LangChain components accept it natively.
    Supports .content, .tool_calls ([{"name","args","id"}]), and .type.
    """

    def __init__(self, body_choice):
        msg = body_choice["message"]
        content    = msg.get("content") or ""
        tool_calls = []
        for tc in msg.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, KeyError):
                args = {}
            tool_calls.append({
                "name": tc["function"]["name"],
                "args": args,
                "id":   tc.get("id", ""),
            })
        if _LC_AIMESSAGE_BASE is not object:
            # Initialise the real AIMessage with content + tool_calls
            super().__init__(content=content, tool_calls=tool_calls)
        else:
            self.content    = content
            self.tool_calls = tool_calls
            self.type       = "ai"


class _BoundChatOpenAI:
    """ChatOpenAI with tools bound — returned by bind_tools(). LCEL-compatible."""

    def __init__(self, parent, tools):
        self._parent = parent
        self._tools  = tools

    def _core_invoke(self, messages):
        return self._parent._invoke_core(messages, tools=self._tools)

    def invoke(self, messages, config=None, **kwargs):
        return self._core_invoke(messages)

    def __call__(self, messages):
        return self._core_invoke(messages)

    def bind_tools(self, tools, **kwargs):
        return _BoundChatOpenAI(self._parent, tools)

    def __or__(self, other):
        try:
            from langchain_core.runnables import RunnableLambda
            return RunnableLambda(self._core_invoke) | other
        except ImportError:
            raise TypeError("langchain_core not available for LCEL chaining")

    def __ror__(self, other):
        try:
            from langchain_core.runnables import RunnableLambda
            return other | RunnableLambda(self._core_invoke)
        except ImportError:
            raise TypeError("langchain_core not available for LCEL chaining")


class ChatOpenAI:
    """
    Drop-in replacement for langchain_openai.ChatOpenAI that routes to LiteLLM.

    Supports:
        llm.invoke(messages)               → AIMessage-like (.content / .tool_calls)
        llm.bind_tools([tool, ...])        → _BoundChatOpenAI
        prompt | llm | parser              → LCEL chain (via RunnableLambda)
    """

    def __init__(self, model=None, temperature=0.7, api_key=None, base_url=None, **kwargs):
        self._model       = model or os.environ.get("MODEL", "gpt-4.1-mini")
        self._temperature = temperature
        self._api_key     = api_key  or os.environ.get("LITELLM_API_KEY", "")
        self._base_url    = (base_url or os.environ.get("LITELLM_BASE_URL", "https://genailab.tcs.in")).rstrip("/")

    def _invoke_core(self, messages, tools=None):
        resolved_model = MODEL_MAP.get(self._model, self._model)
        serialized     = _serialize_lc_messages(messages)
        payload = {
            "model":       resolved_model,
            "messages":    serialized,
            "temperature": self._temperature,
        }
        if tools:
            payload["tools"] = [_lc_tool_schema(t) for t in tools]
        body = _post(f"{self._base_url}/chat/completions", payload, self._api_key)
        return _LangChainAIMessage(body["choices"][0])

    def invoke(self, messages, config=None, **kwargs):
        return self._invoke_core(messages)

    def __call__(self, messages):
        return self._invoke_core(messages)

    def bind_tools(self, tools, **kwargs):
        return _BoundChatOpenAI(self, tools)

    def __or__(self, other):
        """LCEL pipe: llm | parser  →  wraps self as RunnableLambda first."""
        try:
            from langchain_core.runnables import RunnableLambda
            return RunnableLambda(self._invoke_core) | other
        except ImportError:
            raise TypeError("langchain_core not available for LCEL chaining")

    def __ror__(self, other):
        """LCEL pipe: prompt | llm  →  wraps self as RunnableLambda."""
        try:
            from langchain_core.runnables import RunnableLambda
            return other | RunnableLambda(self._invoke_core)
        except ImportError:
            raise TypeError("langchain_core not available for LCEL chaining")

