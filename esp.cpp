#include <Arduino.h>
#include <WiFi.h>
#include <WebSocketsClient.h>
#include "driver/i2s_std.h"
#include <stdint.h>
#include <ArduinoJson.h>
#include <SPI.h>
#include <U8g2lib.h>
#include <vector>

// WiFi credentials for network connection
const char *ssid = "Home 54015 Private";
const char *password = "aimabiet";

// WebSocket server details for communication with web proxy
const char *ws_host = "192.168.1.6";
const uint16_t ws_port = 8080;
const char *ws_path = "/wsesp";

// OLED SPI pins for SSD1309 4-wire hardware SPI display
#define OLED_CLK D8
#define OLED_MOSI D10
#define OLED_RES D0
#define OLED_DC D1
#define OLED_CS D2

constexpr uint8_t SCREEN_WIDTH = 128;
constexpr uint8_t SCREEN_HEIGHT = 64;
constexpr size_t MAX_TEXT_COLUMNS = 32;
constexpr size_t MAX_TEXT_ROWS = 4;

// INMP441 microphone pins: L/R tied to 3.3V for right channel input
#define I2S_SCK 4          // I2S clock pin
#define I2S_WS 5             // I2S word select (LRCLK) pin
#define I2S_SD 6             // I2S data input pin

// I2S audio parameters: 16 kHz, 16-bit mono for voice capture
#define I2S_SAMPLE_RATE 16000
#define I2S_BITS_PER_SAMPLE I2S_DATA_BIT_WIDTH_16BIT
#define I2S_CHANNEL_MODE I2S_SLOT_MODE_MONO

// Buffer size mapped to ~50 milliseconds of audio at 16kHz sample rate
const int I2S_BUFFER_SIZE = 1600;

// Amplification factor to boost microphone input signal amplitude
#define GAIN_BOOSTER 32

i2s_chan_handle_t rx_handle;         // I2S channel handle for reading samples
WebSocketsClient webSocket;          // WebSocket client instance
char i2s_read_buffer[I2S_BUFFER_SIZE]; // Buffer to hold raw audio data

// High-pass filter variable to remove DC offset and low freq noise (~below 50 Hz)
static float last_filtered_sample = 0.0f;
const float HPF_ALPHA = 0.98f;       // Filter coefficient

// OLED display controller for showing WiFi, status, and server messages
U8G2_SSD1309_128X64_NONAME0_F_4W_HW_SPI u8g2(U8G2_R0, OLED_CS, OLED_DC, OLED_RES);

uint8_t contentVisibleColumns = 0;
uint8_t contentVisibleRows = 0;

// Display text management
String currentDisplayText = "";       // Text currently being displayed
String pendingDisplayText = "";      // New text waiting to be displayed
bool isAnimating = false;            // Animation in progress
unsigned long lastAnimationTime = 0;
int animationCharIndex = 0;
int animationCodepointCount = 0;
int animationViewportStartLine = 0;
const int ANIMATION_DELAY_MS = 50;   // Delay between characters

// Display mode: true = translated, false = original (thô)
bool displayTranslated = false;

unsigned long lastI2SErrorLogMs = 0;

void logInfo(const String &msg)
{
  Serial.printf("[%10lu][INFO] %s\n", millis(), msg.c_str());
}

void logWarn(const String &msg)
{
  Serial.printf("[%10lu][WARN] %s\n", millis(), msg.c_str());
}

void logError(const String &msg)
{
  Serial.printf("[%10lu][ERR ] %s\n", millis(), msg.c_str());
}

int utf8CharLen(uint8_t firstByte)
{
  if ((firstByte & 0x80) == 0x00)
  {
    return 1;
  }
  if ((firstByte & 0xE0) == 0xC0)
  {
    return 2;
  }
  if ((firstByte & 0xF0) == 0xE0)
  {
    return 3;
  }
  if ((firstByte & 0xF8) == 0xF0)
  {
    return 4;
  }
  return 1;
}

int countUtf8Codepoints(const String &text)
{
  int count = 0;
  int i = 0;
  const int n = text.length();
  while (i < n)
  {
    int charLen = utf8CharLen((uint8_t)text.charAt(i));
    if (i + charLen > n)
    {
      charLen = 1;
    }
    i += charLen;
    count++;
  }
  return count;
}

String utf8Prefix(const String &text, int codepointCount)
{
  if (codepointCount <= 0)
  {
    return "";
  }

  int i = 0;
  int seen = 0;
  const int n = text.length();
  while (i < n && seen < codepointCount)
  {
    int charLen = utf8CharLen((uint8_t)text.charAt(i));
    if (i + charLen > n)
    {
      charLen = 1;
    }
    i += charLen;
    seen++;
  }
  return text.substring(0, i);
}

void queueDisplayText(const String &text)
{
  if (text.length() == 0)
  {
    return;
  }

  pendingDisplayText = text;
  isAnimating = true;
  animationCharIndex = 0;
  animationCodepointCount = countUtf8Codepoints(text);
  animationViewportStartLine = 0;
}

void useContentFont()
{
  u8g2.setFont(u8g2_font_unifont_t_vietnamese2);
}

void calculateContentCapacity()
{
  useContentFont();

  const uint8_t charWidth = u8g2.getMaxCharWidth();
  const uint8_t charHeight = u8g2.getMaxCharHeight();

  contentVisibleColumns = (charWidth > 0) ? (SCREEN_WIDTH / charWidth) : (uint8_t)MAX_TEXT_COLUMNS;
  contentVisibleRows = (uint8_t)MAX_TEXT_ROWS;

  if (contentVisibleColumns == 0) contentVisibleColumns = 1;
  if (contentVisibleRows == 0) contentVisibleRows = 1;

  logInfo(String("Display capacity cols=") + String(contentVisibleColumns) +
          ", rows=" + String(contentVisibleRows) +
          ", char=" + String(charWidth) + "x" + String(charHeight));
}

void drawStatusScreen(const String &line1, const String &line2 = "")
{
  u8g2.clearBuffer();
  useContentFont();

  const int ascent       = u8g2.getAscent();
  const int descent      = u8g2.getDescent();  // negative in u8g2
  const int firstBaseline = ascent;
  const int lineSpacing   = (SCREEN_HEIGHT + descent - ascent) / (int)(MAX_TEXT_ROWS - 1);
  u8g2.drawUTF8(0, (u8g2_uint_t)firstBaseline, line1.c_str());

  if (line2.length() > 0)
  {
    u8g2.drawUTF8(0, (u8g2_uint_t)(firstBaseline + lineSpacing), line2.c_str());
  }

  u8g2.sendBuffer();
}

void setupDisplay()
{
  SPI.begin(OLED_CLK, -1, OLED_MOSI, OLED_CS);
  u8g2.begin();
  u8g2.setContrast(255);
  u8g2.enableUTF8Print();
  calculateContentCapacity();
  logInfo("u8g2 initialized");
}

// Setup I2S peripheral
void setupI2S()
{

  logInfo("Configuring I2S");

  // I2S RX channel configuration as master capturing from INMP441 mic
  i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
  ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_handle));

  // Standard I2S peripheral configuration: sample rate, mono channel, Philips standard
  i2s_std_config_t std_cfg = {
      .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(I2S_SAMPLE_RATE),
      .slot_cfg = {
          .data_bit_width = I2S_BITS_PER_SAMPLE,
          .slot_bit_width = I2S_SLOT_BIT_WIDTH_16BIT,
          .slot_mode = I2S_CHANNEL_MODE,
          .slot_mask = I2S_STD_SLOT_RIGHT,    // Use right channel due to mic wiring
          .ws_width = I2S_BITS_PER_SAMPLE,
          .ws_pol = false,
          .bit_shift = true                   // Required Philips I2S bit shift setting
      },
      .gpio_cfg = {
          .mclk = I2S_GPIO_UNUSED,
          .bclk = (gpio_num_t)I2S_SCK,
          .ws = (gpio_num_t)I2S_WS,
          .dout = I2S_GPIO_UNUSED,
          .din = (gpio_num_t)I2S_SD}};

  ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_handle, &std_cfg));
  ESP_ERROR_CHECK(i2s_channel_enable(rx_handle));

  logInfo("I2S driver initialized");
}

bool isStarted = false;  // Flag controlling whether audio is being recorded and sent

// WebSocket event handler manages connection state and incoming commands
void webSocketEvent(WStype_t type, uint8_t *payload, size_t length)
{
  switch (type)
  {
  case WStype_DISCONNECTED:
    logWarn("WebSocket disconnected");
    isStarted = false;  // Require explicit esp_start after reconnect
    queueDisplayText("Disconnected!");
    break;

  case WStype_CONNECTED:
    logInfo(String("WebSocket connected: ") + String((char *)payload));
    isStarted = false;  // Start state is controlled by server command
    // Send client type identification
    webSocket.sendTXT("{\"client_type\": \"esp\"}");
    logInfo("Sent client_type=esp handshake");
    queueDisplayText("Connected!");
    break;

  case WStype_TEXT:
  {
    logInfo(String("WS text frame received, bytes=") + String(length));

    StaticJsonDocument<1024> doc;
    DeserializationError error = deserializeJson(doc, payload, length);

    if (error)
    {
      logError(String("deserializeJson failed: ") + String(error.c_str()));
      return;
    }

    String messageType = "";
    if (doc.containsKey("message_type"))
    {
      messageType = doc["message_type"].as<String>();
    }

    if (doc.containsKey("start") || messageType == "esp_start")
    {
      isStarted = true;          // Start sending audio data on command
      queueDisplayText("Recording...");
      logInfo("Received esp_start -> audio streaming enabled");
    }

    if (doc.containsKey("stop") || messageType == "esp_stop")
    {
      isStarted = false;         // Stop sending audio data on command
      queueDisplayText("Stopped.");
      logInfo("Received esp_stop -> audio streaming disabled");
    }
    
    // Handle display mode config
    if (doc.containsKey("esp_display_mode"))
    {
      String mode = doc["esp_display_mode"].as<String>();
      displayTranslated = (mode == "translated");
      logInfo(String("Display mode updated: ") + mode);
    }

    // Handle DisplayText message (for ESP display)
    if (doc.containsKey("message_type") && doc["message_type"] == "DisplayText")
    {
      String incoming = doc["text"].as<String>();
      queueDisplayText(incoming);
      logInfo(String("DisplayText accepted, len=") + String(incoming.length()));
    }

    // Display fallback/error messages from server
    if (doc.containsKey("message_type") && doc["message_type"] == "DisplayError")
    {
      String errText = doc.containsKey("text") ? doc["text"].as<String>() : "[display error]";
      queueDisplayText(errText);
      logWarn(String("DisplayError: ") + errText);
    }

    // Handle ConfigAck message
    if (doc.containsKey("message_type") && (doc["message_type"] == "ConfigAck" || doc["message_type"] == "RuntimeConfig"))
    {
      if (doc.containsKey("esp_display_mode"))
      {
        String mode = doc["esp_display_mode"].as<String>();
        displayTranslated = (mode == "translated");
        logInfo(String("ConfigAck runtime mode: ") + mode);
      }
    }

    if (doc.containsKey("message_type") && doc["message_type"] == "RuntimeConfig")
    {
      String lang = doc.containsKey("target_lang") ? doc["target_lang"].as<String>() : "na";
      bool tr = doc.containsKey("translate") ? doc["translate"].as<bool>() : false;
      logInfo(String("RuntimeConfig synced: translate=") + (tr ? "true" : "false") + ", target=" + lang);
    }

    break;
  }

  case WStype_BIN:
  case WStype_ERROR:
    logError("WebSocket error frame");
    isStarted = false;
    queueDisplayText("WS error");
    break;
  default:
    break;
  }
}

// Wrap UTF-8 text with word-aware breaking: never splits a word across lines.
// Falls back to hard-break only for single words longer than maxCols.
void wrapText(const String &text, std::vector<String> &lines)
{
  lines.clear();

  const int maxCols = contentVisibleColumns > 0 ? contentVisibleColumns : 1;
  String currentLine;
  int    lineLen = 0;
  String currentWord;
  int    wordLen = 0;

  // Flush the accumulated word onto the current display line.
  // If the word is longer than maxCols it is hard-broken before flushing.
  auto flushWord = [&]()
  {
    if (wordLen == 0) return;

    // Hard-break words that exceed the full line width
    while (wordLen > maxCols)
    {
      if (lineLen > 0)
      {
        lines.push_back(currentLine);
        currentLine = "";
        lineLen = 0;
      }
      String piece = utf8Prefix(currentWord, maxCols);
      lines.push_back(piece);
      currentWord = currentWord.substring(piece.length());
      wordLen -= maxCols;
    }

    if (lineLen == 0)
    {
      currentLine = currentWord;
      lineLen     = wordLen;
    }
    else if (lineLen + 1 + wordLen <= maxCols)
    {
      currentLine += ' ';
      currentLine += currentWord;
      lineLen     += 1 + wordLen;
    }
    else
    {
      lines.push_back(currentLine);
      currentLine = currentWord;
      lineLen     = wordLen;
    }
    currentWord = "";
    wordLen     = 0;
  };

  for (int i = 0; i < (int)text.length();)
  {
    int charLen = utf8CharLen((uint8_t)text.charAt(i));
    if (i + charLen > (int)text.length()) charLen = 1;

    char c         = text.charAt(i);
    bool isAscii   = (charLen == 1);
    bool isNewline = isAscii && (c == '\n');
    bool isSpace   = isAscii && (c == ' ' || c == '\t' || c == '\r');

    if (isNewline)
    {
      flushWord();
      lines.push_back(currentLine);
      currentLine = "";
      lineLen     = 0;
      i += charLen;
      continue;
    }
    if (isSpace)
    {
      flushWord();
      i += charLen;
      continue;
    }
    currentWord += text.substring(i, i + charLen);
    wordLen++;
    i += charLen;
  }

  flushWord();
  if (currentLine.length() > 0 || lines.empty())
    lines.push_back(currentLine);
}

// Display text with animation and overflow protection
void displayAnimatedText(String text, int preferredStartLine = -1)
{
  useContentFont();
  u8g2.clearBuffer();

  int maxLines = contentVisibleRows > 0 ? contentVisibleRows : 1;
  if (maxLines < 1)
  {
    maxLines = 1;
  }

  // Distribute rows evenly across 64 px using font metrics so all 4 rows fit.
  const int ascent        = u8g2.getAscent();
  const int descent       = u8g2.getDescent();  // negative in u8g2
  const int firstBaseline = ascent;
  const int lineSpacing   = (maxLines > 1)
      ? (SCREEN_HEIGHT + descent - ascent) / (maxLines - 1)
      : 0;

  std::vector<String> wrappedLines;
  wrapText(text, wrappedLines);

  if (wrappedLines.empty())
  {
    wrappedLines.push_back("");
  }

  int totalLines = wrappedLines.size();
  int startLine = 0;
  if (totalLines > maxLines)
  {
    startLine = totalLines - maxLines;
  }

  if (preferredStartLine >= 0)
  {
    int maxStart = totalLines > maxLines ? (totalLines - maxLines) : 0;
    startLine = preferredStartLine;
    if (startLine < 0)
    {
      startLine = 0;
    }
    if (startLine > maxStart)
    {
      startLine = maxStart;
    }
  }

  for (int row = 0; row < maxLines && (startLine + row) < totalLines; row++)
  {
    const u8g2_uint_t baseline = (u8g2_uint_t)(firstBaseline + row * lineSpacing);
    u8g2.drawUTF8(0, baseline, wrappedLines[startLine + row].c_str());
  }

  u8g2.sendBuffer();
}

// Simple animation update - call this in loop
void updateAnimation()
{
  unsigned long currentTime = millis();

  // Check if there's new text to display
  if (isAnimating && pendingDisplayText != currentDisplayText)
  {
    // Start new animation
    currentDisplayText = pendingDisplayText;
    animationCharIndex = 0;
    animationCodepointCount = countUtf8Codepoints(currentDisplayText);
  }

  // Perform animation step
  if (isAnimating && currentDisplayText.length() > 0)
  {
    if (currentTime - lastAnimationTime >= ANIMATION_DELAY_MS)
    {
      lastAnimationTime = currentTime;

      // Get partial UTF-8-safe text up to current codepoint index
      String partialText = utf8Prefix(currentDisplayText, animationCharIndex + 1);

      std::vector<String> wrappedPreview;
      wrapText(partialText, wrappedPreview);

      int maxLines = contentVisibleRows > 0 ? contentVisibleRows : 1;
      int targetStartLine = 0;
      if ((int)wrappedPreview.size() > maxLines)
      {
        targetStartLine = (int)wrappedPreview.size() - maxLines;
      }

      // Scroll upward gradually (line-by-line) as content approaches screen limit.
      if (targetStartLine > animationViewportStartLine)
      {
        animationViewportStartLine++;
      }

      // Display with overflow protection
      displayAnimatedText(partialText, animationViewportStartLine);

      // Move to next character
      animationCharIndex++;

      // Check if animation is complete
      if (animationCharIndex >= animationCodepointCount)
      {
        isAnimating = false;
      }
    }
  }
}

void setup()
{
  Serial.begin(115200);
  delay(1000);
  logInfo("ESP boot");

  setupDisplay();
  setupI2S();

  drawStatusScreen("Connecting WiFi...");

  // Connect to WiFi and wait until connected before proceeding
  logInfo(String("Connecting WiFi SSID: ") + ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
  }
  logInfo("WiFi connected");
  logInfo(String("IP: ") + WiFi.localIP().toString());

  // Display WiFi connection status and IP on OLED
  drawStatusScreen("WiFi Connected!", WiFi.localIP().toString());

  delay(2000);

  queueDisplayText("Ready.");

  // Initialize WebSocket client with reconnect interval for robustness
  webSocket.beginSSL(ws_host, ws_port, ws_path);
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(5000);
  logInfo(String("WS reconnect interval ms=") + String(5000));
}

void loop()
{
  webSocket.loop();

  // Update animation
  updateAnimation();

  if (webSocket.isConnected() && isStarted)
  {
    size_t bytes_read = 0;

    // Read audio samples from I2S microphone with 100ms timeout
    esp_err_t result = i2s_channel_read(
        rx_handle,
        i2s_read_buffer,
        I2S_BUFFER_SIZE,
        &bytes_read,
        pdMS_TO_TICKS(100));

    if (result == ESP_OK && bytes_read > 0)
    {
      int num_samples = bytes_read / 2;            // 2 bytes per sample (16-bit)
      int16_t *samples = (int16_t *)i2s_read_buffer;

      // Process each sample: apply high-pass filter and gain boost
      for (int i = 0; i < num_samples; i++)
      {
        float current_sample = (float)samples[i];
        float filtered_sample = HPF_ALPHA * last_filtered_sample + (1.0f - HPF_ALPHA) * current_sample;
        last_filtered_sample = filtered_sample;

        int32_t boosted_sample = (int32_t)(filtered_sample * GAIN_BOOSTER);

        // Clamp samples to int16_t range to avoid clipping distortion
        if (boosted_sample > 32767)
        {
          samples[i] = 32767;
        }
        else if (boosted_sample < -32768)
        {
          samples[i] = -32768;
        }
        else
        {
          samples[i] = (int16_t)boosted_sample;
        }
      }

      // Send processed audio buffer over WebSocket as binary data
      webSocket.sendBIN((uint8_t *)i2s_read_buffer, bytes_read);
    }
    else if (result != ESP_OK)
    {
      unsigned long now = millis();
      if (now - lastI2SErrorLogMs >= 1000)
      {
        lastI2SErrorLogMs = now;
        logError(String("I2S read error code=") + String((int)result));
      }
    }
  }
}
