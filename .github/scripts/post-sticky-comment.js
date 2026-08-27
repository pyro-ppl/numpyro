// Copyright Contributors to the Pyro project.
// SPDX-License-Identifier: Apache-2.0

// Posts a benchmark report as a pull request comment, editing the bot's own
// previous comment in place rather than adding a new one on every run.
//
// Shared by both commenting paths: benchmark.yml posts directly when the PR
// comes from a branch in this repository, and benchmark-comment.yml posts on
// its behalf when the PR comes from a fork and the benchmark job therefore had
// a read-only token.

const MARKER = '<!-- numpyro-benchmark-report -->';

// GitHub rejects comment bodies over 65536 characters.
const MAX_BODY = 65000;
const TRUNCATE_TO = 64000;

async function postStickyComment({ github, context, core }, { issue_number, body, runId }) {
  if (!body.startsWith(MARKER)) {
    core.setFailed('Report is missing the expected marker; refusing to post it.');
    return null;
  }

  const runUrl =
    `${context.serverUrl}/${context.repo.owner}/${context.repo.repo}/actions/runs/${runId}`;
  let text = `${body}\n<sub>Produced by [this benchmark run](${runUrl}).</sub>\n`;
  if (text.length > MAX_BODY) {
    text = `${text.slice(0, TRUNCATE_TO)}\n\n…truncated. See [the full report](${runUrl}).\n`;
  }

  const comments = await github.paginate(github.rest.issues.listComments, {
    ...context.repo,
    issue_number,
    per_page: 100,
  });
  const existing = comments.find(
    (c) => c.user.type === 'Bot' && c.body.includes(MARKER),
  );

  if (existing) {
    await github.rest.issues.updateComment({
      ...context.repo,
      comment_id: existing.id,
      body: text,
    });
    core.info(`Updated benchmark comment ${existing.html_url}`);
    return existing.id;
  }

  const { data: created } = await github.rest.issues.createComment({
    ...context.repo,
    issue_number,
    body: text,
  });
  core.info(`Created benchmark comment ${created.html_url}`);
  return created.id;
}

module.exports = postStickyComment;
module.exports.MARKER = MARKER;
