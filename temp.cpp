#include <Arduino.h>
#include <SPI.h>
#include <U8g2lib.h>

#define OLED_CLK D8
#define OLED_MOSI D10
#define OLED_RES D0
#define OLED_DC D1
#define OLED_CS D2

constexpr uint8_t SCREEN_WIDTH = 128;
constexpr uint8_t PROGRESS_X = 0;
constexpr uint8_t PROGRESS_Y = 46;
constexpr uint8_t PROGRESS_WIDTH = SCREEN_WIDTH;
constexpr uint8_t PROGRESS_HEIGHT = 16;
constexpr uint8_t PROGRESS_INSET = 3;
constexpr uint8_t PROGRESS_STEPS = 10;
constexpr unsigned long FRAME_DELAY_MS = 120;

U8G2_SSD1309_128X64_NONAME0_F_4W_HW_SPI u8g2(U8G2_R0, OLED_CS, OLED_DC, OLED_RES);

void setupDisplay()
{
  SPI.begin(OLED_CLK, -1, OLED_MOSI, OLED_CS);
  u8g2.begin();
  u8g2.setContrast(255);
  u8g2.enableUTF8Print();
}

void useVietnameseFont()
{
  u8g2.setFont(u8g2_font_unifont_t_vietnamese2);
}

void drawProgressBar(uint8_t step)
{
  const uint8_t clampedStep = min<uint8_t>(step, PROGRESS_STEPS);
  const uint8_t innerWidth = PROGRESS_WIDTH - (PROGRESS_INSET * 2);
  const uint8_t fillWidth = (innerWidth * clampedStep) / PROGRESS_STEPS;

  u8g2.drawFrame(PROGRESS_X, PROGRESS_Y, PROGRESS_WIDTH, PROGRESS_HEIGHT);
  u8g2.drawBox(
    PROGRESS_X + PROGRESS_INSET,
    PROGRESS_Y + PROGRESS_INSET,
    fillWidth,
    PROGRESS_HEIGHT - (PROGRESS_INSET * 2)
  );
}

void drawScreen(uint8_t progressStep)
{
  u8g2.clearBuffer();
  useVietnameseFont();

  u8g2.drawUTF8(0, 14, "Xin chào");
  u8g2.drawUTF8(0, 30, "XIAO ESP32");
  u8g2.drawUTF8(0, 44, "Tiếng Việt OK");
  drawProgressBar(progressStep);

  u8g2.sendBuffer();
}

void setup()
{
  setupDisplay();
}

void loop()
{
  static uint8_t progressStep = 0;

  drawScreen(progressStep);
  progressStep = (progressStep + 1) % (PROGRESS_STEPS + 1);
  delay(FRAME_DELAY_MS);
}