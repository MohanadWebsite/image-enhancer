# Image Enhancer — Starter Repo (Static + Worker)

هذه الحزمة تحتوي على نسخة أولية من ملفات المشروع لتشغيل خدمة تحسين الصور بالكامل داخل المتصفح باستخدام ONNX Runtime Web.

**محتويات الحزمة**
- `index.html` — واجهة المستخدم (مبسطة).
- `worker.js` — Web Worker الذي يحمل نماذج ONNX ويشغّل المعالجة (tiling, merge, face-restore placeholder).
- `models/` — مجلد placeholder لوضع ملفات النماذج (`.onnx`) (أو ربطها بـR2 عبر الروابط).
- `convert_realesrgan_to_onnx.py` — سكربت تحويل PyTorch -> ONNX (مثال).
- `convert_gfpgan_to_onnx.py` — قابلة للتعديل لتحويل GFPGAN.
- `convert_to_fp16.py` — لتحويل ONNX -> FP16 (اختياري).
- `test_harness.py` — سكربت اختبار بسيط لتشغيل نموذج ONNX على صورة واستخراج زمن التنفيذ.
- `.github/workflows/publish-and-test.yml` — مثال workflow (تخصيص مطلوب للأتمتة).
- `tests/ui-puppeteer-test.js` — اختبار Puppeteer للواجهة (CI).
- `tests/sample.jpg` — صورة اختبار صغيرة.

## تعليمات سريعة (للمبتدئين)
1. افتح المجلد واستخدم سيرفر محلي بسيط:
   ```bash
   python3 -m http.server 8000
   # ثم افتح http://localhost:8000
   ```
2. ضع ملفات النماذج في `models/` (أو اعدل `worker.js` للإشارة إلى روابط R2). أسماء افتراضية:
   - `models/realesrgan_x4.onnx`
   - `models/gfpgan.onnx`
3. افتح الموقع في المتصفح واختر صورة ثم اضغط Start.
4. لتشغيل الاختبارات أو تحويل النماذج راجع السكربتات مع التعليقات داخلها.

---
**ملاحظة:** هذه الملفات تم إعدادها كـstarter template. بعض أجزاء السكربت (مثل تحويل GFPGAN أو أسماء مُدخلات/مخرجات للنماذج) تحتاج تخصيصًا اعتمادًا على إصدار النموذج الذي تستخدمه. إذا أعطيتني روابط ملفات الوزن أو النماذج التي تنوي استخدامها سأخصص السكربتات لك تمامًا.
