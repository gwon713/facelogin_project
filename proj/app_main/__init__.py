__all__=[
    'main_ui',
    'make_widgets',
    'service',
    'main'
]
'''
출석 창에 UI를 만들고 그 안에 관리 버튼을 둔다.
관리 버튼을 누르면 새 창이 뜬다. 관리 창이다.

관리 창에 학생관리 UI를 만들고 그 안에 학번을 적는다. 

영상 스트리밍을 띄우고
동시에 얼굴을 인식한 사각형을 씌워주고,

(loop)학습 버튼을 누르면 그 순간의 사각형 없는 프레임을 찍고,(조건: 얼굴이 있되 1사람만 있어야 함. 1명 초과일 경우 경고)
데이터세트 디렉토리에 ["person_"+학번] 디렉토리를 만들어서 그 안에 현재 시간으로 저장하고

loop가 5번 시행되면 얼굴 스트리밍을 종료한다.
'''

'''
create table student(
id varchar2(20) primary key,
name varchar2(20) not null,
a_time timestamp not null
);
select * from student
'''