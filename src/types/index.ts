export interface Message {
  id: string;
  user_prompt: string;
  model_response: string;
  timestamp: string;
  isStreaming?: boolean;
}

export interface ChatResponse {
  token?: string;
  done?: boolean;
  error?: string;
}

export interface StatusResponse {
  model_loaded: boolean;
  database_connected: boolean;
}
