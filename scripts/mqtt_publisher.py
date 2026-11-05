import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from paho.mqtt import client as mqtt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MQTT_PORT = 8883
CONNECT_TIMEOUT_SECONDS = 15
PUBLISH_TIMEOUT_SECONDS = 10


class MqttRepPublisher:
    def __init__(self, exercise: str) -> None:
        load_dotenv(PROJECT_ROOT / ".env")

        broker = os.getenv("MQTT_BROKER")
        username = os.getenv("MQTT_USERNAME")
        password = os.getenv("MQTT_PASSWORD")
        ca_certificate = Path(
            os.getenv("MQTT_CA_CERT", str(PROJECT_ROOT / "emqxsl-ca.crt"))
        )
        if not broker or not username or not password:
            raise ValueError(
                "MQTT is enabled, but MQTT_BROKER, MQTT_USERNAME, or "
                "MQTT_PASSWORD is missing from the project .env file."
            )
        if not ca_certificate.is_absolute():
            ca_certificate = PROJECT_ROOT / ca_certificate
        if not ca_certificate.is_file():
            raise FileNotFoundError(
                f"MQTT CA certificate not found: {ca_certificate}"
            )

        self.topic = f"fitness/{exercise}/reps"
        self.exercise = exercise
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"fitness-ai-{uuid4().hex[:12]}",
        )
        self.client.username_pw_set(username, password)
        self.client.tls_set(ca_certs=str(ca_certificate))

        connected = threading.Event()
        connection_error: list[str] = []

        def on_connect(client, userdata, flags, reason_code, properties) -> None:
            if reason_code != 0:
                connection_error.append(str(reason_code))
            connected.set()

        self.client.on_connect = on_connect
        self.client.connect(broker, MQTT_PORT, keepalive=60)
        self.client.loop_start()
        if not connected.wait(CONNECT_TIMEOUT_SECONDS):
            self.client.disconnect()
            self.client.loop_stop()
            raise TimeoutError("Timed out connecting to the EMQX MQTT broker.")
        if connection_error:
            self.client.disconnect()
            self.client.loop_stop()
            raise ConnectionError(
                f"EMQX rejected the MQTT connection: {connection_error[0]}"
            )

        print(f"Connected to MQTT broker; publishing to {self.topic}")

    def publish_repetitions(self, count: int) -> None:
        message = json.dumps(
            {
                "exercise": self.exercise,
                "repetitions": count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        result = self.client.publish(self.topic, message, qos=1)
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            raise ConnectionError(
                f"Could not publish repetition count to {self.topic} "
                f"(MQTT error {result.rc})."
            )
        result.wait_for_publish(timeout=PUBLISH_TIMEOUT_SECONDS)
        if not result.is_published():
            raise TimeoutError(f"Timed out publishing a message to {self.topic}.")
        print(f"Published {message} to {self.topic}")

    def close(self) -> None:
        self.client.disconnect()
        self.client.loop_stop()
