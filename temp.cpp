#include <Arduino.h>
#include <SPI.h>
#include <U8g2lib.h>

#define OLED_CLK D8
#define OLED_MOSI D10
#define OLED_RES D0
#define OLED_DC D1
#define OLED_CS D2

constexpr uint8_t SCREEN_WIDTH = 128;
constexpr uint8_t SCREEN_HEIGHT = 64;
constexpr size_t MAX_TEXT_COLUMNS = 32;
constexpr size_t MAX_TEXT_ROWS = 12;

U8G2_SSD1309_128X64_NONAME0_F_4W_HW_SPI u8g2(U8G2_R0, OLED_CS, OLED_DC, OLED_RES);
char textGrid[MAX_TEXT_ROWS][MAX_TEXT_COLUMNS + 1];
uint8_t visibleColumns = 0;
uint8_t visibleRows = 0;

void setupDisplay()
{
  SPI.begin(OLED_CLK, -1, OLED_MOSI, OLED_CS);
  u8g2.begin();
  u8g2.setContrast(255);
  u8g2.enableUTF8Print();
}

void useTestFont()
{
  u8g2.setFont(u8g2_font_5x7_tf);
}

void prepareTextGrid()
{
  for (uint8_t row = 0; row < visibleRows; ++row)
  {
    for (uint8_t col = 0; col < visibleColumns; ++col)
    {
      textGrid[row][col] = 'A' + ((row + col) % 26);
    }
    textGrid[row][visibleColumns] = '\0';
  }
}

void calculateTextCapacity()
{
  useTestFont();

  const uint8_t charWidth = u8g2.getMaxCharWidth();
  const uint8_t charHeight = u8g2.getMaxCharHeight();

  visibleColumns = min<uint8_t>(MAX_TEXT_COLUMNS, SCREEN_WIDTH / charWidth);
  visibleRows = min<uint8_t>(MAX_TEXT_ROWS, SCREEN_HEIGHT / charHeight);

  prepareTextGrid();

  Serial.begin(115200);
  delay(100);
  Serial.println();
  Serial.println("OLED text capacity test");
  Serial.printf("Font: 5x7\n");
  Serial.printf("Char size: %u x %u px\n", charWidth, charHeight);
  Serial.printf("Visible columns: %u\n", visibleColumns);
  Serial.printf("Visible rows: %u\n", visibleRows);
}

void drawScreen()
{
  u8g2.clearBuffer();
  useTestFont();

  const uint8_t lineHeight = u8g2.getMaxCharHeight();

  for (uint8_t row = 0; row < visibleRows; ++row)
  {
    const uint8_t baseline = (row + 1) * lineHeight;
    u8g2.drawStr(0, baseline, textGrid[row]);
  }

  u8g2.sendBuffer();
}

void setup()
{
  setupDisplay();
  calculateTextCapacity();
}

void loop()
{
  drawScreen();
  delay(250);
}