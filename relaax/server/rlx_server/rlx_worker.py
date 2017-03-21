import logging
import traceback

from rlx_config import options

log = logging.getLogger(__name__)


class RLXWorker():

    @classmethod
    def run(cls, socket, address):
        try:
            log.debug('Running worker on connection %s:%d' % address)
            options.protocol.adoptConnection(socket, address)

        except Exception:
            log.error("Something crashed in the worker")
            log.error(traceback.format_exc())
