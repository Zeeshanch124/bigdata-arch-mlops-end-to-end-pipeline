import os
import logging
import requests


class RetrainingTrigger:

    def __init__(self):

        self.github_token = os.getenv(
            'GH_PAT'
        )

        self.repo = (
            'Zeeshanch124/'
            'bigdata-arch-mlops-end-to-end-pipeline'
        )

        self.workflow_id = 'retraining_pipeline.yml'

    def trigger(self):

        logging.warning(
            'Triggering retraining workflow...'
        )

        url = (
            f'https://api.github.com/repos/'
            f'{self.repo}/actions/workflows/'
            f'{self.workflow_id}/dispatches'
        )

        headers = {
            'Accept': 'application/vnd.github+json',
            'Authorization':
                f'Bearer {self.github_token}'
        }

        payload = {
            'ref': 'main'
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload
        )

        if response.status_code == 204:

            logging.info(
                'Retraining workflow '
                'triggered successfully.'
            )

        else:

            logging.error(
                f'Failed to trigger retraining: '
                f'{response.text}'
            )