// 字体加载脚本
import { FONT_CONFIG } from "../config";

document.addEventListener('DOMContentLoaded', () => {
  // 将字体配置注入CSS变量
  const fontFamilyValue = [FONT_CONFIG.fontFamily, ...FONT_CONFIG.fallbackFonts].join(', ');
  document.documentElement.style.setProperty('--font-family', fontFamilyValue);
}); 