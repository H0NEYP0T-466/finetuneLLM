import { useEffect, useRef } from 'react';
import { Message } from '../types';
import './ChatMessage.css';

interface ChatMessageProps {
  message: Message;
}

export const ChatMessage = ({ message }: ChatMessageProps) => {
  const isStreaming = message.isStreaming;
  
  return (
    <div className="chat-message">
      <div className="message-user">
        <span className="message-prompt">$</span> {message.user_prompt}
      </div>
      <div className="message-assistant">
        <span className="message-prompt">&gt;</span> {message.model_response}
        {isStreaming && <span className="cursor">▊</span>}
      </div>
    </div>
  );
};
