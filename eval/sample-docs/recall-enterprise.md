# Recall Enterprise (sample eval document)

Recall scales from a single laptop to a shared team deployment.

Enterprise deployments can use a shared LanceDB object store for the index. Instead of each user maintaining a private on-disk database, the index lives in a central object store that every member of the team reads from, so a document indexed by one person is immediately searchable by everyone.

Role-based access control restricts who may delete documents. By default any user can upload and search, but deletion and re-indexing require a role such as editor or admin. This prevents accidental removal of shared knowledge and keeps the index stable for the whole team.

Audit logs capture every upload and deletion with a timestamp. Each time a document is added or removed, the action is recorded along with the acting user and the time, giving administrators a traceable history of changes to the collective memory.

Enterprise plans also include single sign-on, so authentication can be delegated to an existing identity provider instead of Recall's local JWT. Group membership from the identity provider maps to the roles described above.

For very large corpora, the reranker can be run as a separate service so that the heavier cross-encoder inference does not contend with the web servers. The search route is configured to call that service over the internal network.
