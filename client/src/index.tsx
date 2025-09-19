import {
  AudioClientHelper,
  ConsoleTemplate,
  FullScreenContainer,
  ThemeProvider,
} from '@pipecat-ai/voice-ui-kit';
import { usePipecatClient } from '@pipecat-ai/client-react';
import { StrictMode, useEffect, useRef } from 'react';
import { createRoot } from 'react-dom/client';

//@ts-ignore - fontsource-variable/geist is not typed
import '@fontsource-variable/geist';
//@ts-ignore - fontsource-variable/geist is not typed
import '@fontsource-variable/geist-mono';

const AutoConnect = () => {
  const client = usePipecatClient();
  useEffect(() => {
    // Wait for the first dd > span to show "initialized" status
    const waitForInitialized = () => {
      const ddSpan = document.querySelector('dd > span');
      if (ddSpan && ddSpan.textContent?.trim() === 'initialized') {
        console.log('Client initialized, looking for connect button');
        return true;
      }
      console.log('Client not initialized, waiting for it to be initialized');
      return false;
    };

    const findAndClickConnect = () => {
      const connectButtons = document.querySelectorAll('button[data-slot="button"]');
      for (const button of connectButtons) {
        if (button && button.textContent?.trim() === "Connect") {
          (button as HTMLButtonElement).click();
          console.log('Connect button found and clicked', button);
          return true;
        }
      }
      console.log('Connect button not found');
      return false;
    };

    const checkAndConnect = () => {
      if (waitForInitialized()) {
        if (findAndClickConnect()) {
          return true; // Successfully connected, stop checking
        }
      }
      return false; // Keep checking
    };

    const interval = setInterval(() => {
      if (checkAndConnect()) {
        clearInterval(interval);
      }
    }, 100);

    return () => clearInterval(interval);
  }, [client]);
  return null;
};

createRoot(document.getElementById('root')!).render(
  // @ts-ignore
  <StrictMode>
    <ThemeProvider>
      <FullScreenContainer>
        <AudioClientHelper
          transportType="smallwebrtc"
          connectParams={{ connectionUrl: '/api/offer' }}
        >
          {() => (
            <>
              <AutoConnect />
              <ConsoleTemplate
                transportType="smallwebrtc"
                connectParams={{ connectionUrl: '/api/offer' }}
              />
              <h1>Hello World</h1>
            </>
          )}
        </AudioClientHelper>
      </FullScreenContainer>
    </ThemeProvider>
  </StrictMode>
);
