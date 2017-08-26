# Recurrent Neural Network (RNN)
- Với mạng neural feedforward đơn giản, ta chỉ có thể train được các dữ liệu đơn lẻ mà không thể training được các dữ liệu dạng time series do   trong mạng đơn giản, thứ tự của các neural input không được sắp xếp, có vai trò như nhau.
- RNN là một neural network được điều chỉnh để phù hợp làm việc với dữ liệu dạng sequence, time series.
<p align="center">
  <img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-rolled.png" height="100" width="100">
</p>


<p align= "center">
   <img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png" height="100">
</p>

- Tại mỗi thời điểm t, đầu vào của mỗi recurrent neural là input x(t) và output của thời điểm t-1: y(t-1). Hình vẽ bên dưới mô tả recurrent network được thể hiện theo chiều tăng của thời gian ( được gọi là **unrolled neural through time**)
- Các thuật ngữ, đặc điểm của RNN: (có thể tham khảo tại [1] để có hình minh họa cụ thể hơn)
  - Time step: Một thời điểm t nào đó
  - Tại mỗi time step, ta có một recurrent layer
  - Cell (Memory cell): Một hidden layer trong một recurrent layer 
  - n_neurals: số neural trong hidden layer tại mỗi cell
  - Tại mỗi neural có 2 tập weight: một cho x(t) là Wx, một cho y(t-1) là Wy
  - Số neural trong hidden layer phải bằng số neural trong output.
  - Không giống mạng neural cơ bản sử dụng các tham số khác nhau tại mỗi hidden layer, trong RNN, mỗi recurrent layer có các tham số giống nhau, tức là thực hiện các task như nhau tại mỗi time step, chỉ khác nhau input. Điều này giảm đáng kể số lượng tham số cần phải học.
  - Không nhất thiết tại mỗi recurrent layer phải có các output tương ứng. (xem các kiểu RNN).
- Các kiểu RNN: 
  - Sequence to sequence: input là một chuỗi giá trị theo thời gian, output là một chuỗi giá trị ở đầu ra tương ứng
  - Sequence to vector: input là một chuỗi giá trị theo thời gian, output là một giá trị tại thời điểm t cuối cùng, các giá trị output tại các thời điểm khác bị bỏ qua.
  - vector to sequece: input là một giá trị tại thời điểm ban đầu, output là một chuỗi giá trị đầu ra tại các thời điểm t tiếp theo.
  - delayed sequence to sequece (encoder-decoder)
  
- RNN với Tensorflow : [1]
# Một số loại RNN
- Để train RNN trên các tập đầu vào dạng chuỗi có độ dài lớn, cần rất nhiều vòng lặp, làm cho RNN trở nên rất lớn, dẫn tới hiện tượng vanishing/ exploding gradient (gradient giảm quá chậm hoặc tăng quá nhanh) khiến việc training rất chậm; để tránh việc này, giới hạn số lượng time steps trong quá trình training, giải pháp này được gọi là **truncated back propagate through time (BPTT)**. Tuy nhiên, như vậy sẽ không thể học được các mẫu long-term. Để đảm bảo được việc đó, giải pháp đưa ra là input sequence sẽ chứa cả old data và recent data (VD: monthly data của 5 tháng trước, weekly data của 5 tuần trước, daily data của 5 ngày trước). Tuy nhiên, việc này cũng có giới hạn nếu có một sự kiện ngắn (vd: bầu cử tổng thống) làm ảnh hưởng lớn đến các dự đoán.
- Bên cạnh thời gian training lâu, vấn đề nữa là sau mỗi time step, một vài thông tin sẽ bị mất đi, memory của first input có thể sẽ bị mất sau một số time step. Do đó, ta sử dụng một số loại cell với long-term memory.

## LSTM (Long Short Term Memory)
- Là mô hình long-term memory phổ biến nhất.
- Hiệu quả hơn, hội tụ nhanh hơn RNN thông thường và có thể tìm ra long-term dependencies data.
- Cấu tạo của LSTM cell: (Hình xem trong [1])
  - Giống RNN cell nhưng trạng thái của nó được chia thành 2 vector là h(t) - short term state và c(t) - long term state. Hai trạng thái này được giữ riêng biệt theo mặc định.
  - Ý tưởng: Mạng có thể học xem nên lưu cái gì trong long term state, cái gì nên bỏ và cái gì nên đọc từ nó.
  - x(t) và h(t-1) được truyền qua bộ gồm 4 fully connected layer: 4 lớp recurrent layer khác nhau, trong đó :
      - Lớp chính thường có vai trò phân tích current input x(t) và trạng thái trước đó h(t-1) (short term). Trong RNN, output đi thẳng ra ngoài nhưng trong LSTM, một phần output được lưu trong long term state.
      - Ba lớp còn lại là gate controller để điều khiển các cổng, mỗi lớp sử dụng logistic activation function, output nằm trong khoảng (0,1).
          - forget gate: Điều khiển xóa một số phần trong long term state
          - input gate: Điều khiển thêm một số phần (của output của lớp chính) vào long term state.
          - output gate: Điều khiển xem phần nào của long term state nên được đọc và output ra tại thời điểm hiện tại cho h(t) và y(t).
- Tóm lại, LSTM học cách phát hiện xem phần nào là thông tin quan trọng (input state), lưu nó vào long term memory, lưu trữ nó đến khi nào vẫn còn cần thiết(forget state) và trích xuất nó khi nào cần.

## Peephole Connections
- Là một biến thể của LSTM được tạo thêm các kết nối gọi là peephole connections. Previous long term state c(t-1) được thêm vào như input của controller của forget state và input state; current long term state c(t) được thêm vào như input của controller của output state.

## GRU ( Gated Recurrent Unit)
- Là một biến thể khác của LSTM. Nó là một phiên bản được đơn giản hóa của LSTM cell, và có hiệu quả tương đương.
- Các phần chính được đơn giản hóa:
  - Cả 2 vector trạng thái được gộp vào h(t).
  - Một gate controller điều khiển cả forget gate và input gate. Nếu gate controller output ra 1 thì input gate được mở và forget gate bị đóng. Nếu gate controller output ra 0 thì ngược lại. Hay nói cách khác, khi nào memory phải được lưu trữ thì vị trí nó sẽ được lưu vào phải được xóa đi trước.
  - Không có output gate, vector trạng thái đầy đủ sẽ được output tại mỗi time step. Tuy nhiên, có một gate controller mới điều khiển thành phần nào của previous state sẽ được hiển thị lên main layer.

- Nhờ LSTM, GRU mà RNN được sử dụng phổ biến trong những năm gấn đây, cụ thể là trong xử lý ngôn ngữ tự nhiên NLP.
  
# Về phần NLP và bài tập củng cố các kiến thức trên xem trong [1]  
  
# Tài liệu tham khảo:
1. [Hands on Machine Learning with scikit learn and tensorflow concepts, tools and techniques to build intelligient](http://gen.lib.rus.ec/search.php?req=Hands+on+Machine+Learning+with+scikit+learn+and+tensorflow&open=0&res=25&view=simple&phrase=0&column=def) - Aurelien Geron - Chương 14
2. 
