# Báo cáo Khảo sát và Đề xuất Nghiên cứu: Grounding kết hợp Learning to Reject và Explainable AI

## Tóm tắt
Báo cáo này trình bày một khảo sát toàn diện về các hướng nghiên cứu tiềm năng trong lĩnh vực Visual Grounding, đặc biệt tập trung vào việc tích hợp khả năng từ chối dự đoán (Learning to Reject) và giải thích được (Explainable AI - XAI). Chúng tôi phân tích các mô hình Grounding hiện đại, xác định một bài báo cơ sở (base paper) có mã nguồn, chỉ ra những hạn chế của nó và đề xuất các hướng cải tiến cụ thể, bao gồm việc sử dụng các kỹ thuật định lượng độ bất định (Uncertainty Quantification) và các benchmark phù hợp để đánh giá.

## 1. Khảo sát các mô hình Grounding hiện đại (SOTA)
Lĩnh vực **Visual Grounding** (hay Referring Expression Comprehension - REC) đang chứng kiến sự phát triển mạnh mẽ, chuyển dịch từ các mô hình chuyên biệt sang các mô hình nền tảng (Foundation Models) có khả năng open-set. Các mô hình này có khả năng xác định và khoanh vùng các đối tượng trong hình ảnh dựa trên mô tả ngôn ngữ tự nhiên [1].

Bảng 1: Tổng quan các mô hình Grounding SOTA và khả năng liên quan

| Mô hình | Đặc điểm nổi bật | Khả năng XAI/Rejection |
| :--- | :--- | :--- |
| **Grounding DINO** (2024) | Mô hình SOTA cho open-set object detection, kết hợp DINO Transformer với ngôn ngữ [2]. | Thấp (chủ yếu dựa trên confidence score của detector). |
| **Florence-2** (Microsoft) | Mô hình đa nhiệm mạnh mẽ, hỗ trợ nhiều tác vụ thị giác-ngôn ngữ, bao gồm grounding [3]. | Trung bình (cấu trúc encoder-decoder cho phép truy xuất thông tin). |
| **Rex-Thinker** (2025) | Sử dụng **Chain-of-Thought (CoT)** để suy luận từng bước, cho phép giải thích quá trình ra quyết định [4]. | **Cao** (giải thích qua các bước Planning-Action-Summarization). |
| **SafeGround** (2026) | Tập trung vào **Uncertainty Calibration** cho GUI Grounding, cung cấp cơ chế từ chối dựa trên kiểm soát tỷ lệ phát hiện sai (FDR) [5]. | **Cao** (có cơ chế từ chối dựa trên FDR control và định lượng độ bất định). |

## 2. Base Paper Đề xuất: Rex-Thinker (IDEA-Research, 2025)
Chúng tôi đề xuất chọn **Rex-Thinker: Grounded Object Referring via Chain-of-Thought Reasoning** [4] làm bài báo cơ sở (Base Paper) cho nghiên cứu của bạn. Bài báo này được công bố bởi IDEA-Research vào năm 2025 và có mã nguồn mở trên GitHub [6]. Rex-Thinker là một lựa chọn phù hợp vì nó hội tụ đủ các yếu tố quan trọng mà bạn quan tâm:

*   **Grounding**: Mô hình này xây dựng dựa trên Grounding DINO để tạo ra các ứng viên đối tượng (candidate objects) và sau đó sử dụng một Mô hình Ngôn ngữ Lớn Đa phương thức (MLLM) để lọc và xác định đối tượng cuối cùng.
*   **Explainable AI (XAI)**: Rex-Thinker sử dụng cấu trúc suy luận Chain-of-Thought (CoT) rõ ràng, bao gồm các giai đoạn Planning, Action và Summarization. Điều này cho phép người dùng hiểu được quá trình mô hình đưa ra quyết định, làm tăng tính minh bạch và khả năng giải thích.
*   **Learning to Reject**: Mô hình đã tích hợp cơ chế từ chối khi không tìm thấy đối tượng phù hợp trong danh sách ứng viên, hoặc khi không có đối tượng nào khớp với mô tả ngôn ngữ [4].
*   **Mã nguồn**: Mã nguồn của Rex-Thinker đã được công bố chính thức, tạo điều kiện thuận lợi cho việc tái tạo và mở rộng nghiên cứu [6].

## 3. Phân tích Limitations của Rex-Thinker và Hướng Cải thiện
Mặc dù Rex-Thinker là một bước tiến đáng kể, nó vẫn tồn tại một số hạn chế có thể được khai thác để phát triển các hướng nghiên cứu mới:

1.  **Phụ thuộc vào Detector bên ngoài**: Rex-Thinker phụ thuộc hoàn toàn vào Grounding DINO để tạo ra các ứng viên đối tượng. Nếu Grounding DINO bỏ lỡ đối tượng mục tiêu (recall thấp), MLLM sẽ không thể tìm thấy nó, dẫn đến giới hạn về hiệu suất tổng thể của hệ thống [4].
2.  **Uncertainty chưa được định lượng (Quantified)**: Cơ chế từ chối của Rex-Thinker chủ yếu mang tính "logic" (từ chối nếu không có sự khớp), nhưng chưa có sự tính toán và định lượng rõ ràng về độ bất định (Uncertainty Estimation) của dự đoán. Điều này khác biệt so với các phương pháp như trong SafeGround, nơi độ bất định được định lượng một cách có hệ thống [5].
3.  **Chi phí tính toán cao**: Việc thực hiện suy luận CoT cho từng ứng viên đối tượng có thể tốn kém về mặt tính toán, đặc biệt với các hình ảnh có nhiều đối tượng hoặc các mô tả phức tạp.
4.  **Thiếu tính hiệu chỉnh (Calibration)**: Confidence score của câu trả lời cuối cùng từ Rex-Thinker có thể không được hiệu chỉnh tốt, dẫn đến tình trạng "overconfident" (quá tự tin) ngay cả khi dự đoán sai. Điều này làm giảm độ tin cậy của mô hình trong các ứng dụng thực tế.

## 4. Đề xuất Hướng Nghiên cứu Cải thiện
Để khắc phục những hạn chế trên và tạo ra một mô hình Visual Grounding đáng tin cậy và có khả năng giải thích cao hơn, chúng tôi đề xuất kết hợp các ưu điểm của **SafeGround** (Uncertainty Calibration) vào cấu trúc của **Rex-Thinker** (CoT Reasoning). Hướng nghiên cứu này có thể bao gồm các điểm sau:

*   **Cải thiện cơ chế từ chối (Rejection)**: Thay vì chỉ dựa vào việc MLLM nói "No", hãy tích hợp các kỹ thuật định lượng độ bất định như **Conformal Prediction** [7] hoặc **Evidential Deep Learning** [8] vào từng bước suy luận của CoT. Điều này sẽ cho phép mô hình không chỉ từ chối khi không chắc chắn mà còn cung cấp một thước đo định lượng về mức độ không chắc chắn đó.
*   **Hiệu chỉnh độ tin cậy (Uncertainty Calibration)**: Áp dụng các phương pháp hiệu chỉnh độ tin cậy (ví dụ: Temperature Scaling, Isotonic Regression) cho confidence score của mô hình để đảm bảo rằng xác suất dự đoán phản ánh chính xác độ tin cậy thực tế của mô hình. Điều này sẽ giúp mô hình "biết cái nó không biết" và tránh đưa ra các dự đoán quá tự tin khi sai [9].
*   **Tối ưu hóa chi phí tính toán**: Nghiên cứu các phương pháp để giảm chi phí tính toán của CoT, ví dụ như sử dụng kỹ thuật pruning các ứng viên không tiềm năng sớm trong quá trình suy luận hoặc áp dụng các mô hình CoT nhẹ hơn.

## 5. Benchmark và Metric đề xuất để đánh giá
Để đánh giá hiệu quả của mô hình cải tiến, chúng tôi đề xuất sử dụng các benchmark và metric sau:

*   **Benchmark cho độ chính xác Grounding cơ bản**: Sử dụng các tập dữ liệu tiêu chuẩn như **RefCOCO, RefCOCO+, và RefCOCOg** để đánh giá khả năng grounding chính xác của mô hình trong các điều kiện thông thường [10].
*   **Benchmark cho Out-of-Distribution (OOD) và Rejection**: Sử dụng các tập dữ liệu được thiết kế đặc biệt cho OOD như **OODBench** [11] hoặc tạo ra các mẫu OOD bằng cách sử dụng các câu lệnh không có đối tượng thực tế (Null-expression) hoặc các hình ảnh không liên quan. Điều này sẽ đánh giá khả năng của mô hình trong việc nhận diện và từ chối các trường hợp nằm ngoài phân phối dữ liệu huấn luyện.
*   **Metric cho Uncertainty Calibration và Rejection**: 
    *   **Expected Calibration Error (ECE)**: Đo lường sự khác biệt giữa độ tin cậy dự đoán và độ chính xác thực tế của mô hình [9].
    *   **False Discovery Rate (FDR)**: Kiểm soát tỷ lệ các dự đoán sai được chấp nhận, đặc biệt quan trọng trong các ứng dụng nhạy cảm [5].
    *   **Rejection Rate**: Tỷ lệ các trường hợp mô hình từ chối đưa ra dự đoán, cùng với độ chính xác trên các trường hợp được chấp nhận.

## Kết luận
Việc kết hợp Visual Grounding với Learning to Reject và Explainable AI mở ra một hướng nghiên cứu đầy hứa hẹn để xây dựng các hệ thống AI không chỉ mạnh mẽ mà còn đáng tin cậy và minh bạch. Bằng cách cải tiến Rex-Thinker với các kỹ thuật định lượng độ bất định và hiệu chỉnh độ tin cậy, chúng ta có thể phát triển một mô hình có khả năng tự nhận biết giới hạn của mình, từ chối các dự đoán không chắc chắn và cung cấp lời giải thích rõ ràng cho các quyết định của nó. Điều này sẽ góp phần quan trọng vào việc triển khai AI an toàn và hiệu quả hơn trong các ứng dụng thực tế.

## Tài liệu tham khảo
[1] L. Xiao et al., "Towards Visual Grounding: A Survey," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2025. [https://ieeexplore.ieee.org/abstract/document/11235566/]
[2] S. Liu et al., "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection," *arXiv preprint arXiv:2303.05499*, 2024. [https://arxiv.org/abs/2303.05499]
[3] Microsoft Florence-2. [https://github.com/microsoft/Florence-2]
[4] Q. Jiang et al., "Rex-Thinker: Grounded Object Referring via Chain-of-Thought Reasoning," *arXiv preprint arXiv:2506.04034*, 2025. [https://arxiv.org/abs/2506.04034]
[5] Q. Wang et al., "SafeGround: Know When to Trust GUI Grounding Models via Uncertainty Calibration," *arXiv preprint arXiv:2602.02419*, 2026. [https://arxiv.org/abs/2602.02419]
[6] IDEA-Research/Rex-Thinker GitHub Repository. [https://github.com/IDEA-Research/Rex-Thinker]
[7] A. G. Vovk et al., "Conformal Prediction for Reliable Machine Learning," *Journal of Machine Learning Research*, 2005.
[8] A. Sensoy et al., "Evidential Deep Learning to Quantify Uncertainty in Neural Networks," *Advances in Neural Information Processing Systems*, 2018.
[9] C. Guo et al., "On Calibration of Modern Neural Networks," *International Conference on Machine Learning*, 2017.
[10] B. Yu et al., "RefCOCO, RefCOCO+, RefCOCOg: A Dataset for Referring Expression Comprehension," *arXiv preprint arXiv:1611.09783*, 2016.
[11] L. Lin et al., "OODBench: Out-of-Distribution Benchmark for Large Vision-Language Models," *arXiv preprint arXiv:2602.18094*, 2026. [https://arxiv.org/abs/2602.18094]
