from fastapi.testclient import TestClient

from lean_swarm.api.app import create_app


def main() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/simulate",
        json={
            "seed_document": "A survey suggests voters are divided on the new reform package.",
            "question": "Will support increase over the next month?",
            "rounds": 4,
        },
    )
    print(response.json()["report"])


if __name__ == "__main__":
    main()

