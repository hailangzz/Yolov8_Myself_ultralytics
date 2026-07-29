import requests
import base64


class SAM3VideoClient:


    def __init__(self, url, model):
        self.url=url
        self.model=model


    def encode(self,path):

        with open(path,"rb") as f:
            return base64.b64encode(
                f.read()
            ).decode()


    def init_video(self,frames):

        data={

            "model":self.model,

            "frames":
            [
                self.encode(x)
                for x in frames
            ],

            "start_frame_index":0
        }


        r=requests.post(
            self.url+"/v1/video/init",
            json=data
        )

        return r.json()



    def prompt_point(
        self,
        session_id,
        x,y
    ):

        data={

          "session_id":session_id,

          "model":self.model,

          "frame_index":0,

          "points":[
              [x,y]
          ],

          "point_labels":[1],

          "obj_id":1
        }


        r=requests.post(
            self.url+"/v1/video/prompt",
            json=data
        )

        return r.json()



    def propagate(
        self,
        session_id,
        start=0,
        end=None
    ):

        data={

        "session_id":session_id,

        "model":self.model,

        "start_frame":start,

        "end_frame":end

        }


        r=requests.post(
            self.url+"/v1/video/propagate",
            json=data
        )

        return r.json()


client=SAM3VideoClient(
    "http://172.16.50.229:9000",
    "segment_anything_3_video"
)


session=client.init_video(
    [
     "1.jpg",
     "2.jpg",
     "3.jpg"
    ]
)


sid=session["session_id"]


client.prompt_point(
    sid,
    300,
    200
)


task=client.propagate(
    sid,
    0,
    100
)


print(task)