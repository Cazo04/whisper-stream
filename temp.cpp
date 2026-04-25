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
constexpr size_t MAX_TEXT_ROWS = 4;
constexpr uint8_t TEST_ROWS = 4;

U8G2_SSD1309_128X64_NONAME0_F_4W_HW_SPI u8g2(U8G2_R0, OLED_CS, OLED_DC, OLED_RES);
String textGrid[MAX_TEXT_ROWS];
uint8_t visibleColumns = 0;
uint8_t visibleRows = 0;

// Kích thước mảng này là đủ màn
const char *kVietnameseGlyphs[] = {
  "àáảãạăắằẳẵâấầẩẫậ",
  "âấầẩẫậđèéẻẽ-----",
  "êếềểễệìíỉĩò-----",
  "óỏõọôốồổỗộ------"
};

void setupDisplay()
{
  SPI.begin(OLED_CLK, -1, OLED_MOSI, OLED_CS);
  u8g2.begin();
  u8g2.setContrast(255);
  u8g2.enableUTF8Print();
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

String utf8Prefix(const char *text, uint8_t codepointCount)
{
  String result;
  uint8_t consumed = 0;
  const size_t length = strlen(text);
  size_t index = 0;

  while (index < length && consumed < codepointCount)
  {
    const int charLen = utf8CharLen(static_cast<uint8_t>(text[index]));
    result += String(text).substring(index, index + charLen);
    index += charLen;
    consumed++;
  }

  return result;
}

void useTestFont()
{
  u8g2.setFont(u8g2_font_unifont_t_vietnamese2);
}

void prepareTextGrid()
{
  for (uint8_t row = 0; row < visibleRows; ++row)
  {
    textGrid[row] = utf8Prefix(kVietnameseGlyphs[row % TEST_ROWS], visibleColumns);
  }
}

void calculateTextCapacity()
{
  useTestFont();

  const uint8_t charWidth = u8g2.getMaxCharWidth();
  const uint8_t charHeight = u8g2.getMaxCharHeight();

  visibleColumns = MAX_TEXT_COLUMNS;
  visibleRows = min<uint8_t>(MAX_TEXT_ROWS, SCREEN_HEIGHT / charHeight);

  prepareTextGrid();

  Serial.begin(115200);
  delay(100);
  Serial.println();
  Serial.println("OLED text capacity test");
  Serial.printf("Font: unifont_t_vietnamese2\n");
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
    u8g2.drawUTF8(0, baseline, textGrid[row].c_str());
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