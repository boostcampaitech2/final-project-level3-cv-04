import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
  ComponentProps,
} from "streamlit-component-lib"
import React, { ReactNode } from "react"

interface State {}

class TinyWebrtc extends StreamlitComponentBase<State> {
  public state = {}

  private pc: RTCPeerConnection | undefined
  private videoRef: React.RefObject<HTMLVideoElement>

  constructor(props: ComponentProps) {
    super(props)

    this.pc = undefined
    this.videoRef = React.createRef()
  }

  public componentDidUpdate = () => {
    if (this.props.args["answer"]) {
      const answer = new RTCSessionDescription(this.props.args["answer"])
      this.processAnswer(answer)
    }
  }

  public render = (): ReactNode => {
    return (
      <div>
        <button onClick={this.start}>Start</button>
        <button onClick={this.stop}>Stop</button>
        <video
          ref={this.videoRef}
          autoPlay
          playsInline
          onCanPlay={() => Streamlit.setFrameHeight()}
        />
      </div>
    )
  }

  private createPeerConnection = () => {
    const config: RTCConfiguration = {}
    const pc = new RTCPeerConnection(config)

    // connect audio / video
    pc.addEventListener("track", (evt) => {
      if (evt.track.kind === "video") {
        const videoElement = this.videoRef.current
        if (videoElement != null) {
          videoElement.srcObject = evt.streams[0]
        } else {
          console.warn("Video element is not mounted.")
        }
      }
    })

    return pc
  }

  private negotiate = (pc: RTCPeerConnection) => {
    return pc
      .createOffer()
      .then(function (offer) {
        return pc.setLocalDescription(offer)
      })
      .then(function () {
        // wait for ICE gathering to complete
        return new Promise<void>(function (resolve) {
          if (pc.iceGatheringState === "complete") {
            resolve()
          } else {
            const checkState = () => {
              if (pc.iceGatheringState === "complete") {
                pc.removeEventListener("icegatheringstatechange", checkState)
                resolve()
              }
            }
            pc.addEventListener("icegatheringstatechange", checkState)
          }
        })
      })
      .then(function () {
        const offer = pc.localDescription

        if (offer == null) {
          console.error("Offer is null.")
          return
        }

        // Offer is created!
        const offerJson = offer.toJSON()
        console.log("Send offer SDP to Python process: ", offerJson)
        Streamlit.setComponentValue({
          offerJson,
        })
      })
  }

  private processAnswer = (answer: RTCSessionDescription) => {
    if (this.pc == null) {
      console.error("this.pc is not initialized yet.")
      return
    }

    this.pc.setRemoteDescription(answer)
  }

  private start = () => {
    const pc = this.createPeerConnection()

    const constraints: MediaStreamConstraints = {
      audio: false,
      video: true,
    }

    navigator.mediaDevices.getUserMedia(constraints).then(
      (stream) => {
        stream.getTracks().forEach(function (track) {
          pc.addTrack(track, stream)
        })
        return this.negotiate(pc)
      },
      function (err) {
        alert("Could not acquire media: " + err)
      }
    )

    this.pc = pc
  }

  private stop = () => {
    const pc = this.pc
    this.pc = undefined

    if (pc == null) {
      return
    }

    // close transceivers
    if (pc.getTransceivers) {
      pc.getTransceivers().forEach(function (transceiver) {
        if (transceiver.stop) {
          transceiver.stop()
        }
      })
    }

    // close local audio / video
    pc.getSenders().forEach(function (sender) {
      sender.track?.stop()
    })

    // close peer connection
    setTimeout(function () {
      pc.close()
    }, 500)
  }
}

export default withStreamlitConnection(TinyWebrtc)
