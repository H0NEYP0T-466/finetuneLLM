import { useState, useEffect, useRef } from 'react';
import type { Message } from '../types';
import { apiService } from '../services/api';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { LoadingIndicator } from './LoadingIndicator';
import './ChatContainer.css';

export const ChatContainer = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    const initializeChat = async () => {
      try {
        // Check model status
        const status = await apiService.getStatus();
        setIsModelLoaded(status.model_loaded);

        if (!status.model_loaded) {
          setError('Model not loaded. Please ensure a .gguf model file is in the backend/model directory.');
          setIsLoading(false);
          return;
        }

        // Load chat history
        const history = await apiService.getMessages(20);
        setMessages(history);
        setIsLoading(false);
      } catch (err) {
        console.error('Failed to initialize chat:', err);
        setError('Failed to connect to the server. Please ensure the backend is running.');
        setIsLoading(false);
      }
    };

    initializeChat();
  }, []);

  const handleSendMessage = async (prompt: string) => {
    if (!isModelLoaded || isStreaming) return;

    const newMessage: Message = {
      id: Date.now().toString(),
      user_prompt: prompt,
      model_response: '',
      timestamp: new Date().toISOString(),
      isStreaming: true,
    };

    setMessages((prev) => [...prev, newMessage]);
    setIsStreaming(true);

    await apiService.sendMessage(
      prompt,
      (token: string) => {
        setMessages((prev) => {
          const updated = [...prev];
          const lastMessage = updated[updated.length - 1];
          if (lastMessage) {
            lastMessage.model_response += token;
          }
          return updated;
        });
      },
      () => {
        setMessages((prev) => {
          const updated = [...prev];
          const lastMessage = updated[updated.length - 1];
          if (lastMessage) {
            lastMessage.isStreaming = false;
          }
          return updated;
        });
        setIsStreaming(false);
      },
      (errorMsg: string) => {
        console.error('Error streaming message:', errorMsg);
        setIsStreaming(false);
        setError(errorMsg);
      }
    );
  };

  if (isLoading) {
    return <LoadingIndicator message="Initializing chat..." />;
  }

  if (error && !isModelLoaded) {
    return (
      <div className="error-container">
        <div className="error-message">
          <span className="error-icon">⚠</span>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-container">
      <div className="chat-header">
        <span className="header-text">FineTuneLLM Terminal</span>
        <span className={`status-indicator ${isModelLoaded ? 'online' : 'offline'}`}>
          {isModelLoaded ? '● Online' : '● Offline'}
        </span>
      </div>
      <div className="chat-messages" ref={chatContainerRef}>
        {messages.length === 0 ? (
          <div className="welcome-message">
            <p>Welcome to FineTuneLLM Terminal</p>
            <p>Type a message to start chatting...</p>
          </div>
        ) : (
          messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))
        )}
        <div ref={messagesEndRef} />
      </div>
      <ChatInput onSendMessage={handleSendMessage} disabled={isStreaming || !isModelLoaded} />
    </div>
  );
};
