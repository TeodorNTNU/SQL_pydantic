import React, { useState, useEffect, useRef } from 'react';
import '@chatscope/chat-ui-kit-styles/dist/default/styles.min.css';
import {
  MainContainer,
  ChatContainer,
  MessageList,
  Message,
  MessageInput,
  TypingIndicator
} from '@chatscope/chat-ui-kit-react';

const App = () => {
  // State to manage user input
  const [input, setInput] = useState('');
  // State to manage chat messages
  const [messages, setMessages] = useState([]);
  // State to manage typing indicator
  const [typing, setTyping] = useState(false);
  // Ref to manage the WebSocket connection
  const ws = useRef(null);
  // State to manage reconnection attempts
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const maxReconnectAttempts = 5;

  // Function to setup WebSocket connection
  const setupWebSocket = () => {
    if (ws.current) {
      ws.current.close(); // Ensure any existing WebSocket is closed
    }

    ws.current = new WebSocket('ws://127.0.0.1:8000/ws/chat/');
    let ongoingStream = null; // To track the ongoing stream's ID

    ws.current.onopen = () => {
      console.log('WebSocket connected!');
      setReconnectAttempts(0); // Reset reconnect attempts on successful connection
    };

    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      let sender = data.name;

      // Handle different types of events from the WebSocket
      if (data.event === 'on_parser_start') {
        // When a new stream starts
        ongoingStream = { id: data.run_id, content: '' };
        setMessages((prevMessages) => [...prevMessages, { message: '', sender: sender, direction: 'incoming', id: data.run_id }]);
      } else if (data.event === 'on_parser_stream' && ongoingStream && data.run_id === ongoingStream.id) {
        // During a stream, appending new chunks of data
        setMessages((prevMessages) =>
          prevMessages.map((msg) =>
            msg.id === data.run_id ? { ...msg, message: msg.message + data.data.chunk } : msg
          )
        );
      }

      setTyping(false); // Hide typing indicator after processing message
    };

    ws.current.onerror = (event) => {
      console.error('WebSocket error observed:', event);
    };

    ws.current.onclose = (event) => {
      console.log(`WebSocket is closed now. Code: ${event.code}, Reason: ${event.reason}`);
      if (event.code !== 1000) { // 1000 indicates normal closure
        handleReconnect();
      }
    };
  };

  // Function to handle reconnection attempts with exponential backoff
  const handleReconnect = () => {
    if (reconnectAttempts < maxReconnectAttempts) {
      setReconnectAttempts((prevAttempts) => prevAttempts + 1);
      let timeout = Math.pow(2, reconnectAttempts) * 1000; // Exponential backoff
      console.log(`Attempting to reconnect in ${timeout / 1000} seconds...`);
      setTimeout(() => {
        setupWebSocket(); // Attempt to reconnect
      }, timeout);
    } else {
      console.log('Max reconnect attempts reached, not attempting further reconnects.');
    }
  };

  // Effect hook to setup and cleanup the WebSocket connection
  useEffect(() => {
    setupWebSocket(); // Setup WebSocket on component mount

    return () => {
      if (ws.current && ws.current.readyState === WebSocket.OPEN) {
        ws.current.close(); // Close WebSocket on component unmount
      }
    };
  }, []); // Empty dependency array ensures this runs only once on mount and unmount

  // Handler for sending messages
  const handleSend = (message) => {
    const userMessage = { message: message, sender: 'user', direction: 'outgoing' };
    setMessages((prevMessages) => [...prevMessages, userMessage]);

    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      setTyping(true); // Show typing indicator while waiting for response
      ws.current.send(JSON.stringify({ message })); // Send message through WebSocket
    } else {
      console.error('WebSocket is not connected.');
    }
  };

  return (
    <div className="App">
      <div style={{ position: 'relative', height: '800px', width: '700px' }}>
        <MainContainer>
          <ChatContainer>
            <MessageList typingIndicator={typing ? <TypingIndicator content="ChatGPT is typing..." /> : null}>
              {messages.map((message, i) => (
                <Message key={i} model={{
                  message: message.message,
                  sentTime: "just now",
                  sender: message.sender,
                  direction: message.direction
                }} />
              ))}
            </MessageList>
            <MessageInput placeholder="Type your message here" onSend={handleSend} />
          </ChatContainer>
        </MainContainer>
      </div>
    </div>
  );
};

export default App;
