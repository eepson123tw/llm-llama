import { Check, Copy } from "lucide-react";

import { JSONValue, Message } from "ai";
import Image from "next/image";
import { Button } from "../button";
import ChatAvatar from "./chat-avatar";
import Markdown from "./markdown";
import { useCopyToClipboard } from "./use-copy-to-clipboard";

interface ChatMessageImageData {
  type: "image_url";
  image_url: {
    url: string;
  };
}

// This component will parse message data and render the appropriate UI.
function ChatMessageData( {messageData}:{messageData:string} ) {

  if (messageData) {

    const base64Regex = /"url":\s*"([^"]+)"/;
    const match = messageData.match(base64Regex);
    let base64Data = "";
    if (match && match[1]) {
      base64Data = match[1];
      console.log('Extracted base64 data:', base64Data);
    } else {
      console.log('No base64 data found.');
    }

    return (
      <div className="rounded-md max-w-[200px] shadow-md">
        <Image
          src={base64Data}
          width={0}
          height={0}
          sizes="100vw"
          style={{ width: "100%", height: "auto" }}
          alt=""
        />
      </div>
    );
  }
  return null;
}

export default function ChatMessage(chatMessage: Message) {
  const { isCopied, copyToClipboard } = useCopyToClipboard({ timeout: 2000 });
  const hasContent = chatMessage.content.includes('url')
  const data = hasContent ? chatMessage.content : ''
   const base64Regex = /"message":\s*"([^"]+)"/;
   let content = ''
   const match = chatMessage.content.match(base64Regex);
    if (match && match[1]) {
      content = match[1];
      console.log('Extracted base64 data:', content);
    } else {
      console.log('No base64 data found.');
    }
  return (
    <div className="flex items-start gap-4 pr-5 pt-5">
      <ChatAvatar role={chatMessage.role} />
      <div className="group flex flex-1 justify-between gap-2">
        <div className="flex-1 space-y-4">
          {hasContent && (
            <ChatMessageData messageData={data} />
          )}
          <Markdown content={ content||chatMessage.content} />
        </div>
        <Button
          onClick={() => copyToClipboard(chatMessage.content)}
          size="icon"
          variant="ghost"
          className="h-8 w-8 opacity-0 group-hover:opacity-100"
        >
          {isCopied ? (
            <Check className="h-4 w-4" />
          ) : (
            <Copy className="h-4 w-4" />
          )}
        </Button>
      </div>
    </div>
  );
}
