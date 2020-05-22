**CNN:**  
- 이미지 내 모든 픽셀들이 모두 연관 관계를 가지고 있는것은 아님
- 이미지 내 픽셀들의 특징
    - 지역적으로 관련이 높으며(locally related), 멀리 떨어진 경우 수치적 관련성이 매우 낮음
    - 높은 지역적 관계(local relationship), 공간적인 상관관계(spatial correlation)
    - 하지만....
- 기존 NN의 경우 overfitting될 위험이 높으며(parameter증가) local relationship을 고려하지 않는 문제

# 특징

- 이미지 인식에서 기존 NN의 단점을 보완한 알고리즘
- 위치에 관계없이 시각적 특징을 추출하기 위해 제안
- local connection(지역적 연결) + shared weight(파라미터 공유)
- convolution, pooling(set)

## convolutional layer
- 지정된 크기의 filter를 통해 linear combination 을 진행하여 feature map 생성
    - fillter != receptive field
- 이후 feature map이 activation을 통과하여 activation map을 형성
    - 여러개의 activation map이 합쳐져 하나의 이미지 블록(block)으로 모음
    - 블록의 depth = 직전 layer에서 사용한 filter 갯수
- 이 흐름이 convolutional layer의 flow

## padding
- feature map의 크기 감소를 예방하기 위한 방법
- 두 가지 이유
    - 결과의 차원 감소를 방지하기 위해(same)
    - 경계 부분의 계산을 진행하기 위해(valid)
    
## pooling layer
- activation map의 크기를 효과적으로 줄임(with subsampling)
- 중요정보유지
- overfitting 방지
- 계산량 감소
- 허나, layer가 많아지면 많아질수록 pooling layer는 감소하는 추세

## FC layer (fully connected layer)
- conv, pooling layer를 지날수록 image의 구체적 정보만 남게됨
    - spare image
- 위 문제를 해결하기 위해 사용하는 layer -> fully connected layer
    - 이미지에 포함된 고차원의 숨은 표현 추출(hidden representation)
    - 추출된 정보를 바탕으로 다양한 특징을 확보하기 위해
    - 앞에서 배운 기본 NN의 형식과 같음
