"""Opt-in FUTURE queue mixin, not attached to any running controller.

An explicit archive_completed_small job flag is required. Existing native views
cannot be relabelled, and every postprocessing failure keeps training_completed.
The legacy compactor is bypassed BEFORE it can mmap a guarded legacy archive.
"""
import os

from src.utils import research_checkpoint_archival as archival


class ProspectiveSmallArchivalMixin:
    """Place before an existing ResearchQueue subclass in a new controller MRO.

    This is not a deployed entry point. The future controller must bind this
    module in its own source/readiness/execution proof before using the mixin.
    """
    def compact_completed_job(self,job,record):
        if not job.get('archive_completed_small',False):
            return super().compact_completed_job(job,record)
        from scripts.research_queue import output_dir,artifacts_for,timestamp
        run=output_dir(job['argv'])
        record.update(training_completed=True,failure_phase='checkpoint_compaction',
            checkpoint_postprocess_kind=archival.OWNER)
        self.persist()
        try:
            if not job.get('compact_completed'):raise ValueError('Archival requires completed optimizer compaction')
            # Detection is metadata-only, before calling the old direct-mmap
            # compactor. A partial archived transaction must recover via its
            # journal, never through that old compactor.
            archived=os.path.lexists(run/archival.JOURNAL) or any(
                archival.read((run/role).resolve()/'.research_checkpoint.json').get('checkpoint_archival')
                for role in ('best','last') if (run/role).exists())
            if archived:
                archival.eligibility(run)
                record['compaction']=archival.read(run/'checkpoint_compaction.json')
                self.persist()
            else:
                super().compact_completed_job(job,record)
            activation=self.spec.get('trained_small_archival_activation')
            result=archival.publish_completed_checkpoint(run,activation=activation,
                reserve_gb=self.spec.get('reserve_gb',120))
            checked=archival.validate_completed_archival(run)
            if result!=checked:raise ValueError('Durable completed archival revalidation differs')
            record.update(checkpoint_archival=checked,artifacts=artifacts_for(job),failure_phase=None)
            self.persist()
        except Exception as error:
            # Reuse the base queue's explicit retry status; a novel unknown
            # status could otherwise bypass its --retry-failed guard.
            record.update(training_completed=True,failure_phase='checkpoint_compaction',
                status='training_completed_compaction_failed',finished=timestamp(),
                error_type=type(error).__name__,error=str(error))
            self.persist()
            raise
