using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using Unity.Mathematics;
#if UNITY_EDITOR
using UnityEditor;
#endif
using UnityEngine;
using static ClothDynamics.GPUClothDynamics;

namespace ClothDynamics
{
    [DefaultExecutionOrder(15300)] //When using Final IK

    public class ClothObjectGPU : GPUClothBase
    {
        int _resolution;
        internal int2 _indexOffset;
        internal ClothSolverGPU m_solver;
        float _particleDiameter;
        private MaterialPropertyBlock _mpb;
        private MeshRenderer _mr;
        private SkinnedMeshRenderer _smr;
        private MeshFilter _mf;
        private bool _generatedMesh = false;
        private List<int> _attachedIndices;
        //private bool _applyAllIndices = false;

        private List<Vector2Int> _connectionInfo = null;
        private List<int> _connectedVerts = null;
        [SerializeField] public bool _weldVertices = true;
        [SerializeField] public bool _sewEdges = false;
        [SerializeField] public bool _fixDoubles = false;
        private bool _showWeldEdges = false;
        private HashSet<int> _dupIngoreList;
        private List<BendingConstraintStruct> _extraBendingConstraints;
        private List<Edge> _weldEdges;
        internal bool _weldDone = false;
        private bool _addConstraints = true;

        [Tooltip("Objects like buttons can be added here and follow the vertices of the cloth.")]
        public Transform[] _followObjects;
        [Tooltip("Constraint objects can be added here.")]
        public Transform[] _attachedObjects;

        static int _foundObjectsInScene = 0;
        static int _objectsCounter = 0;

        private Shader _shader = null;
        private bool _useTransferData = false;

#if UNITY_EDITOR
        [ReadOnly]
#endif
        [SerializeField] internal int _clothId;
        internal GPUClothDynamicsV2 _dynamics;

        [Header("Proxy Skin")]
        [Tooltip("This curve controls the weighting of the skinned MeshProxy.")]
        [SerializeField] public AnimationCurveData _weightsCurve;
        [Tooltip("This is the tolerance distance for the vertices that affect the weighting.")]
        [SerializeField] public float _weightsToleranceDistance = 0.05f;
        [Tooltip("This scales the weighting.")]
        [SerializeField] public float _scaleWeighting = 50;
        [Tooltip("This is the minimal radius between the connected vertices.")]
        [SerializeField] public float _minRadius = 0.001f;
        [Tooltip("This skin is automatically generated the first time you run the cloth sim. It will be saved on disk to be reused for the next run to increase loading time.")]
        [SerializeField] public UnityEngine.Object _skinPrefab;

        public bool _useGarmentMesh = false;
        public float _garmentSeamLength = 0.02f;

        public bool _applyTransformAtExport = false;

        private MaterialPropertyBlock _mpbHD;
        private ComputeShader _clothSolver;
        private int _numGroups_VerticesHD;
        private int _maxVerts_ID = Shader.PropertyToID("_maxVerts");
        private int _positions_ID = Shader.PropertyToID("_positions");
        private int _normalsBuffer_ID = Shader.PropertyToID("_normalsBuffer");
        private int _mpb_positionsBuffer_ID = Shader.PropertyToID("positionsBuffer");
        private int _mpb_normalsBuffer_ID = Shader.PropertyToID("normalsBuffer");
        private int _connectionInfoBuffer_ID = Shader.PropertyToID("_connectionInfoBuffer");
        private int _connectedVertsBuffer_ID = Shader.PropertyToID("_connectedVertsBuffer");
        private int _worldToLocalMatrixHD_ID = Shader.PropertyToID("_worldToLocalMatrixHD");
        private int _localToWorldMatrix_ID = Shader.PropertyToID("_localToWorldMatrix");

        private int _csNormalsKernel;
        private int _skinningHDKernel;
        private ComputeBuffer _connectionInfoTetBuffer;
        private ComputeBuffer _connectedVertsTetBuffer;
        private ComputeBuffer _startVertexBuffer;
        private ComputeBuffer _vertexBlendsBuffer;
        private ComputeBuffer _bonesStartMatrixBuffer;

        private WaitForSeconds _waitForSeconds = new WaitForSeconds(1);

        private Vector3 _savePos;
        private Vector3 _saveSca;
        private Quaternion _saveRot;
        private Transform parent;
        private Transform _originalParent;

        private ComputeBuffer _tempBuffer;

        private void Awake()
        {
            _originalParent = this.transform.parent;
            var mr = GetComponent<Renderer>();
            if (mr != null)
            {
                for (int i = 0; i < mr.materials.Length; i++)
                {
                    var material = mr.materials[i];
                    //var cacheShader = material.shader;
                    if (material.shader.name.ToLower().Contains("cloth"))
                    {
                        //    material.shader = Resources.Load("Shaders/V2/ClothShaderV2", typeof(Shader)) as Shader; // Shader.Find("ClothDynamics/ClothSurfaceShader");
                        //    _shader = material.shader;
                        //    if (cacheShader != material.shader) material.SetFloat("_EmissiveIntensity", 0);
                        //}
                        //if (_shader != null) material.shader = _shader;

                        if (_tempBuffer == null) _tempBuffer = new ComputeBuffer(1, sizeof(float) * 3);
                        material.SetBuffer("positionsBuffer", _tempBuffer);
                        material.SetBuffer("normalsBuffer", _tempBuffer);
                    }
                    //material.EnableKeyword("USE_BUFFERS");
                    //if (_useTransferData) material.EnableKeyword("USE_TRANSFER_DATA");
                }
            }
        }

        public void Init(GPUClothDynamicsV2 dynamics, int resolution, ClothSolverGPU solver, bool generated = false)
        {
            _dynamics = dynamics;
            m_solver = solver;
            _resolution = resolution;
            _generatedMesh = generated;
        }

        IEnumerator Start()
        {
            if (_clothSolver == null) _clothSolver = Resources.Load("Shaders/Compute/PBDClothSolver") as ComputeShader;

            if (_foundObjectsInScene == 0)
            {
                var list = FindObjectsOfType<ClothObjectGPU>();
                if (list != null) _foundObjectsInScene = list.Length;
            }
            var animator = this.GetComponentInParent<Animator>();
            bool animState = false;
            if (animator != null)
            {
                Debug.Log(this.name);
                animState = animator.enabled;
                animator.enabled = false;
            }
            //_version = 2;

            if (_dynamics == null)
            {
                _dynamics = FindObjectOfType<GPUClothDynamicsV2>();
            }

            if (_dynamics._solver._localSpace)
            {
                if (this.GetComponent<ClothSkinningGPU>() && this.transform.parent && (this.transform.parent.position.sqrMagnitude != 0 || this.transform.parent.rotation != Quaternion.identity))
                {
                    _savePos = this.transform.parent.position;
                    _saveRot = this.transform.parent.rotation;
                    _saveSca = this.transform.parent.localScale;
                    this.transform.parent.position = Vector3.zero;
                    this.transform.parent.rotation = Quaternion.identity;
                    this.transform.parent.localScale = Vector3.one;
                    parent = this.transform.parent;
                }
            }

            m_solver = _dynamics._solver;
            Extensions.CleanupList(ref _dynamics._clothList);
            if (!_dynamics._clothList.Contains(this.gameObject)) _dynamics._clothList.Add(this.gameObject);

            Extensions.CleanupList(ref _attachedObjects);

            //_mr = this.GetComponent<MeshRenderer>();
            //_smr = this.GetComponent<SkinnedMeshRenderer>();
            //SetMatProperties();

            //yield return _waitForSeconds;
            for (int i = 0; i < 2; i++) yield return null;

            var mesh = this.GetComponent<MeshFilter>() ? this.GetComponent<MeshFilter>().sharedMesh : this.GetComponent<SkinnedMeshRenderer>() ? this.GetComponent<SkinnedMeshRenderer>().sharedMesh : null;
            if (mesh == null || mesh.vertexCount < 1) yield break;

            if (_weldVertices && !_weldDone) { GPUClothDynamics.WeldVerticesOld(mesh, out _); _weldDone = true; }

            _mr = this.GetComponent<MeshRenderer>();
            _smr = this.GetComponent<SkinnedMeshRenderer>();

            if (_mr != null)
            {
                var b = _mr.bounds;
                b.size *= 1000;//TODO
                _mr.bounds = b;
            }
            else if (_smr)
            {
                var b = _smr.bounds;
                b.size *= 1000;//TODO
                _smr.bounds = b;
            }

            float4x4 transformMatrix = this.transform.localToWorldMatrix;

            //mesh.vertexBufferTarget |= GraphicsBuffer.Target.Vertex;
            //var vBuffer = mesh.GetVertexBuffer(0);
            //var temp = new float[mesh.vertexCount * 10];
            //vBuffer.GetData(temp);
            //var verts = new Vector3[temp.Length/10];
            //for (int i = 0; i < verts.Length; i++)
            //{
            //    verts[i] = new Vector3(temp[i * 10 + 0], temp[i * 10 +1], temp[i * 10 + 2]);
            //}
            //var iBuffer = mesh.GetIndexBuffer();
            //int[] tris = new int[iBuffer.count];
            //iBuffer.GetData(tris);

            var positions = mesh.vertices;
            var indices = mesh.triangles;
            var colors = mesh.colors;

            _particleDiameter = math.length(positions[0] - positions[1]) * _dynamics._globalSimParams.particleDiameterScalar;

            //if (_applyAllIndices) m_attachedIndices.AddRange(indices);

            //if (!_usePreCache || _preCacheFile == null) //TODO

            bool useProxy = _useMeshProxy && _meshProxy != null;
            _objBuffers = new ObjectBuffers[useProxy ? 2 : 1];
            if (useProxy)
            {
                for (int i = 0; i < _objBuffers.Length; i++)
                {
                    _objBuffers[i] = new ObjectBuffers();
                }
            }

            if (_addConstraints)
            {
                CalcConnections(mesh, 0, _objBuffers[0]);

                if (_sewEdges && _fixDoubles)
                    ExtraConstraints(mesh, positions);
            }

            _clothId = m_solver._meshDataList.Count;
            _indexOffset = m_solver.AddCloth(_dynamics, this.gameObject, mesh, transformMatrix, _particleDiameter);

            ApplyTransform(ref positions, transformMatrix);

            m_solver.AddMasses(_addConstraints ? 1 : 0);

            if (_addConstraints)
            {
                if (_generatedMesh)
                    GenerateStretch(positions);
                else
                    AddDistanceConstraints(positions, indices, colors);
                GenerateAttach(positions);
            }
            //GenerateBending(indices);
            _objectsCounter++;

#if UNITY_2021_2_OR_NEWER
            var skinning = this.GetComponent<GPUSkinning>();
            if (skinning) _useTransferData = this.GetComponent<GPUSkinning>()._useTransferData;
#endif
            SetClothShader(_mr ? _mr : _smr);

            _mpb = new MaterialPropertyBlock();
            if (_objectsCounter >= _foundObjectsInScene) SetMatProperties();

            _meshObjects = m_solver._collisionMeshes._meshObjects;

            List<Mesh> meshList = new List<Mesh>();
            meshList.Add(mesh);
            if (useProxy)
            {
                var mf = _meshProxy.GetComponent<MeshFilter>();
                Mesh meshHD = null;
                if (mf != null)
                    meshHD = mf.mesh;
                else
                {
                    var smr = _meshProxy.GetComponent<SkinnedMeshRenderer>();
                    if (meshHD == null && smr != null)
                    {
                        meshHD = smr.sharedMesh;
                        mf = _meshProxy.gameObject.AddComponent<MeshFilter>();
                        mf.sharedMesh = meshHD;
                        var mr = _meshProxy.GetComponent<MeshRenderer>();
                        if (mr == null) mr = _meshProxy.gameObject.AddComponent<MeshRenderer>();
                        Material[] materials = smr.sharedMaterials;
                        if (smr) DestroyImmediate(smr);
                        mr.sharedMaterials = materials;
                        if (mf.sharedMesh.subMeshCount < mr.sharedMaterials.Length)
                        {
                            var mats = mr.sharedMaterials;
                            Array.Resize(ref mats, mf.sharedMesh.subMeshCount);
                            mr.sharedMaterials = mats;
                        }
                    }
                }
                if (meshHD != null) meshList.Add(meshHD);
                else { useProxy = false; if (_meshProxy != null) _meshProxy.SetActive(false); }

                for (int i = 1; i < _objBuffers.Length; i++)
                {
                    //if (!_usePreCache || _preCacheFile == null)
                    CalcConnections(meshList[i], i, _objBuffers[i]);
                }
            }
            else if (_meshProxy != null) _meshProxy.SetActive(false);

            //if (_usePreCache) PreCacheClothData(meshList);

            SetupSkinningHD(debugMode: false, mesh);     

            StartCoroutine(DelayUpdate(indices));
            
            _finishedLoading = true;

            for (int i = 0; i < 2; i++) yield return null;

            var mask = this.GetComponent<ProjectionMask>();

            if (mask != null && mask._copy != null)
            {
                mask._copy.transform.parent = this.transform.parent;
            }

            if (animator != null)
            {
                animator.enabled = animState;
            }
        }

        bool _delayUpdateFinished = false;

        private void FixedUpdate()
        {
           if(_delayUpdateFinished) SetupFollowObjects();
        }

        IEnumerator DelayUpdate(int[] indices)
        {
            //Debug.Log("_objectsCounter " + _objectsCounter + " _foundObjectsInScene " + _foundObjectsInScene);
            if (_objectsCounter >= _foundObjectsInScene)
            {
                //m_solver.UpdateBuffers();
                yield return null;
                m_solver.UpdateBuffers();
            }

            yield return null;

            if (_dynamics._solver._localSpace)
            {
                if (this.GetComponent<ClothSkinningGPU>() && parent && (_savePos.sqrMagnitude != 0 || _saveRot != Quaternion.identity))
                {
                    parent.position = _savePos;
                    parent.rotation = _saveRot;
                    parent.localScale = _saveSca;
                }
            }

            SetupFollowObjects(indices);

            _delayUpdateFinished = true;

        }

        private void SetClothShader(Renderer mr)
        {
            if (mr != null)
            {
                //print("SetClothShader " + mr.gameObject.name + " mats " + mr.materials.Length);
                for (int i = 0; i < mr.materials.Length; i++)
                {
                    var material = mr.materials[i];
                    var cacheShader = material.shader;
                    //Debug.Log(this.name + " " + material.shader.name + " " + material.shader.name.ToLower().Contains("cloth"));
                    if (_shader == null && !material.shader.name.ToLower().Contains("cloth"))
                    {
                        material.shader = Resources.Load("Shaders/V2/ClothShaderV2", typeof(Shader)) as Shader; // Shader.Find("ClothDynamics/ClothSurfaceShader");
                        _shader = material.shader;
                        if (cacheShader != material.shader) material.SetFloat("_EmissiveIntensity", 0);
                    }
                    if (_shader != null) material.shader = _shader;

                    if (_tempBuffer == null) _tempBuffer = new ComputeBuffer(1, sizeof(float) * 3);
                    material.SetBuffer("positionsBuffer", _tempBuffer);
                    material.SetBuffer("normalsBuffer", _tempBuffer);

                    material.EnableKeyword("USE_BUFFERS");
                    if (_useTransferData) material.EnableKeyword("USE_TRANSFER_DATA");
                }
            }
        }

        public void SetMatProperties()
        {
            if (_mpb == null) _mpb = new MaterialPropertyBlock();
            if (_mpb != null)
            {
                _mpb.SetBuffer(_mpb_positionsBuffer_ID, /*m_solver._localSetup ? m_solver.positions2 :*/ m_solver._positions);
                _mpb.SetBuffer(_mpb_normalsBuffer_ID, m_solver._normals);
                SetCustomProperties(_mpb);
                if (_mr) _mr.SetPropertyBlock(_mpb);
                else if (_smr) _smr.SetPropertyBlock(_mpb);
            }
        }

        internal override void SetCustomProperties(MaterialPropertyBlock mpb)
        {
            mpb.SetMatrix("worldToLocalMatrix", this.transform.worldToLocalMatrix);
            mpb.SetVector("rotation", QuatToVec(this.transform.rotation));
            if (m_solver != null && m_solver._meshDataList != null && m_solver._meshDataList.Count > 0)
            {
                var meshData = m_solver._meshDataList[_clothId];
                mpb.SetInt("prevNumParticles", meshData.prevNumParticles);
            }
        }

        internal override ComputeBuffer GetPositionsBuffer()
        {
            return m_solver?._positions;
        }

        internal override ComputeBuffer GetNormalsBuffer()
        {
            return m_solver?._normals;
        }

        public Vector4 QuatToVec(Quaternion rot)
        {
            Vector4 rotVec;
            rotVec.x = rot.x;
            rotVec.y = rot.y;
            rotVec.z = rot.z;
            rotVec.w = rot.w;
            return rotVec;
        }

        void ApplyTransform(ref Vector3[] positions, float4x4 transform)
        {
            for (int i = 0; i < positions.Length; i++)
            {
                positions[i] = math.mul(transform, new float4(positions[i], 1.0f)).xyz;
            }
        }

        private void CalcConnections(Mesh pMesh, int index = 0, ObjectBuffers obj = null)
        {
            //var pMesh = _mesh;
            Vector3[] vertices = pMesh.vertices;
            int[] faces = pMesh.triangles;
            int lastCount = 0;// verts.Count;
            List<Vector2Int> connectionInfo = new List<Vector2Int>();
            List<int> connectedVerts = new List<int>();
            Dictionary<Vector3, List<int>> dictTris = new Dictionary<Vector3, List<int>>();

            for (int f = 0; f < faces.Length; f += 3)
            {
                if (dictTris.ContainsKey(vertices[faces[f]]))
                {
                    var list = dictTris[vertices[faces[f]]];
                    list.Add(lastCount + faces[f + 1]);
                    list.Add(lastCount + faces[f + 2]);
                }
                else
                {
                    dictTris.Add(vertices[faces[f]], new List<int>(new[] {
                                                lastCount + faces [f + 1],
                                                lastCount + faces [f + 2]
                                            }));
                }
                if (dictTris.ContainsKey(vertices[faces[f + 1]]))
                {
                    var list = dictTris[vertices[faces[f + 1]]];
                    list.Add(lastCount + faces[f + 2]);
                    list.Add(lastCount + faces[f]);
                }
                else
                {
                    dictTris.Add(vertices[faces[f + 1]], new List<int>(new[] {
                                                lastCount + faces [f + 2],
                                                lastCount + faces [f]
                                            }));
                }
                if (dictTris.ContainsKey(vertices[faces[f + 2]]))
                {
                    var list = dictTris[vertices[faces[f + 2]]];
                    list.Add(lastCount + faces[f]);
                    list.Add(lastCount + faces[f + 1]);
                }
                else
                {
                    dictTris.Add(vertices[faces[f + 2]], new List<int>(new[] {
                                                lastCount + faces [f],
                                                lastCount + faces [f + 1]
                                            }));
                }
            }
            int currentNumV = vertices.Length;
            int maxVertexConnection = 0;
            float[] minDistArray = new float[currentNumV];
            float minDistance = float.MaxValue;
            float maxDistance = 0;
            for (int n = 0; n < currentNumV; n++)
            {
                if (!dictTris.ContainsKey(vertices[n])) continue;
                var list = dictTris[vertices[n]];
                int start = connectedVerts.Count;
                float dist = float.MaxValue;
                for (int i = 0; i < list.Count; i++)
                {
                    connectedVerts.Add(list[i]);
                    float d = Vector3.Distance(vertices[n], vertices[list[i]]);
                    if (n != list[i] || d > float.Epsilon)
                        dist = Mathf.Min(dist, d);
                }
                int end = connectedVerts.Count;
                maxVertexConnection = Mathf.Max(maxVertexConnection, end - start);
                connectionInfo.Add(new Vector2Int(start, end));
                minDistArray[n] = dist;
                minDistance = Mathf.Min(dist, minDistance);
                maxDistance = Mathf.Max(dist, maxDistance);
            }

            if (obj != null)
            {
                obj.connectionInfoBuffer = new ComputeBuffer(connectionInfo.Count, sizeof(int) * 2);
                obj.connectionInfoBuffer.SetData(connectionInfo.ToArray());
                obj.connectedVertsBuffer = new ComputeBuffer(connectedVerts.Count, sizeof(int));
                obj.connectedVertsBuffer.SetData(connectedVerts.ToArray());

                if (obj.normalsBuffer != null) obj.normalsBuffer.Release();
                obj.normalsBuffer = new ComputeBuffer(vertices.Length, sizeof(float) * 3);
                obj.normalsBuffer.SetData(pMesh.normals);

                if (obj.positionsBuffer != null) obj.positionsBuffer.Release();
                obj.positionsBuffer = new ComputeBuffer(vertices.Length, sizeof(float) * 3);
                obj.positionsBuffer.SetData(vertices);
            }

            if (index == 0)
            {
                _connectionInfo = connectionInfo;
                _connectedVerts = connectedVerts;

                Vector4[] projectedPosData = new Vector4[vertices.Length];
                for (int n = 0; n < projectedPosData.Length; n++)
                {
                    projectedPosData[n] = vertices[n];
                    projectedPosData[n].w = minDistArray[n] * 0.5f;
                }
                //_projectedPositionsBuffer = new ComputeBuffer(vertices.Length, sizeof(float) * 4);
                //_projectedPositionsBuffer.SetData(projectedPosData);
            }
        }

        private void ExtraConstraints(Mesh pMesh, Vector3[] positions)
        {
            var aMesh = pMesh;// this.GetComponent<MeshFilter>().sharedMesh;
            var verts = aMesh.vertices;
            SortedSet<int> newDupVerts = new SortedSet<int>();
            _dupIngoreList = new HashSet<int>();

            Dictionary<int, int> duplicateHashTable = new Dictionary<int, int>();

            for (int i = 0; i < verts.Length; i++)
            {
                if (!duplicateHashTable.ContainsKey(i))
                {
                    duplicateHashTable.Add(i, i);
                }
                else
                {
                    var index = duplicateHashTable[i];
                    newDupVerts.Add(index);
                    newDupVerts.Add(i);
                    _dupIngoreList.Add(i);
                }
            }

            _extraBendingConstraints = new List<BendingConstraintStruct>();

            if (_showWeldEdges) _weldEdges = new List<Edge>();
            HashSet<int> noDupConList = new HashSet<int>();
            List<int> wingCorners = new List<int>();

            //var array = newDupVerts.ToArray();
            //int count = newDupVerts.Count;
            var enumerator = newDupVerts.GetEnumerator();
            while (enumerator.MoveNext())
            //for (int i = 0; i < count; i += 2)
            {
                int index = enumerator.Current;
                enumerator.MoveNext(); // used again to jump two index
                //int index = array[i];// newDupVerts.ElementAt(i); //ElementAt is too slow
                if (_connectedVerts != null)
                {
                    int n = index;
                    Vector2Int info = _connectionInfo[n];
                    int start = info.x;
                    int end = info.y;

                    int connectedIndex = 0;
                    noDupConList.Clear();
                    for (int c = start; c < end; ++c)
                    {
                        int conId = _connectedVerts[c];
                        if (newDupVerts.Contains(conId))
                        {
                            connectedIndex = conId;
                        }
                        else
                        {
                            noDupConList.Add(conId);
                        }
                    }

                    if (_showWeldEdges)
                    {
                        _weldEdges.Add(new Edge(index, connectedIndex));
                    }

                    n = connectedIndex;
                    info = _connectionInfo[n];
                    start = info.x;
                    end = info.y;
                    wingCorners.Clear();
                    for (int c = start; c < end; ++c)
                    {
                        int conId = _connectedVerts[c];
                        if (noDupConList.Contains(conId))
                        {
                            wingCorners.Add(conId);
                        }
                    }

                    if (wingCorners.Count < 2) continue;

                    int4 indices = 0;
                    indices[0] = wingCorners[0];
                    indices[1] = wingCorners[1];
                    indices[2] = index;
                    indices[3] = connectedIndex;

                    var bendingConstraint = new BendingConstraintStruct();
                    bendingConstraint.index0 = indices[0];
                    bendingConstraint.index1 = indices[1];
                    bendingConstraint.index2 = indices[2];
                    bendingConstraint.index3 = indices[3];
                    Vector3 p0 = positions[indices[0]];
                    Vector3 p1 = positions[indices[1]];
                    Vector3 p2 = positions[indices[2]];
                    Vector3 p3 = positions[indices[3]];

                    Vector3 n1 = (Vector3.Cross(p2 - p0, p3 - p0)).normalized;
                    Vector3 n2 = (Vector3.Cross(p3 - p1, p2 - p1)).normalized;

                    float d = Vector3.Dot(n1, n2);
                    d = Mathf.Clamp(d, -1.0f, 1.0f);
                    bendingConstraint.restAngle = Mathf.Acos(d);

                    _extraBendingConstraints.Add(bendingConstraint);
                }
            }
        }

        private void AddDistanceConstraints(Vector3[] positions, int[] indices, Color[] baseColors)
        {
            float DistanceBetween(int idx1, int idx2)
            {
                return math.length(positions[idx1] - positions[idx2]);
            };

            // use a set to get unique edges
            HashSet<Edge> edgeSet = new HashSet<Edge>(new EdgeComparer());

            if (_sewEdges)
            {
                for (int n = 0; n < _connectionInfo.Count; n++)
                {
                    Vector2Int info = _connectionInfo[n];
                    int start = info.x;
                    int end = info.y;
                    for (int c = start; c < end; ++c)
                    {
                        //if (!_dupIngoreList.Contains(n) && !_dupIngoreList.Contains(_connectedVerts[c]))
                        if (!_fixDoubles || !_dupIngoreList.Contains(n))
                            edgeSet.Add(new Edge(n, _connectedVerts[c]));
                    }
                }
                if (_fixDoubles)
                {
                    _dupIngoreList.Clear();
                    _dupIngoreList = null;
                }
            }
            else
            {
                int trisLength = indices.Length / 3;
                for (int i = 0; i < trisLength; i++)
                {
                    edgeSet.Add(new Edge(indices[i * 3 + 0], indices[i * 3 + 1]));
                    edgeSet.Add(new Edge(indices[i * 3 + 0], indices[i * 3 + 2]));
                    edgeSet.Add(new Edge(indices[i * 3 + 1], indices[i * 3 + 2]));
                }
            }

            //_numDistanceConstraints = edgeSet.Count;
            //_distanceConstraints = new DistanceConstraintStruct[_numDistanceConstraints];
            //int j = 0;
            int idx1, idx2;
            foreach (Edge e in edgeSet)
            {
                idx1 = e.startIndex;
                idx2 = e.endIndex;

                var restLength = DistanceBetween(idx1, idx2);
                if (idx1 < baseColors.Length && idx2 < baseColors.Length)
                    restLength = _useGarmentMesh && baseColors[idx1].g + baseColors[idx2].g > 1.5f && restLength > _garmentSeamLength ? _garmentSeamLength : restLength;

                m_solver.AddStretch(_indexOffset.x + idx1, _indexOffset.x + idx2, restLength);

                //_distanceConstraints[j].edge = edge;
                //var dist = (_positions[edge.startIndex] - _positions[edge.endIndex]).magnitude;
                //if (edge.startIndex < baseColors.Length && edge.endIndex < baseColors.Length)
                //    _distanceConstraints[j].restLength = _useGarmentMesh && baseColors[edge.startIndex].g + baseColors[edge.endIndex].g > 1.5f && dist > _garmentSeamLength ? _garmentSeamLength : dist;
                //else
                //    _distanceConstraints[j].restLength = dist;
                //j++;
            }

            //Debug.Log("_numDistanceConstraints " + _numDistanceConstraints);
            m_solver.UpdateStretch();

        }

        void GenerateStretch(Vector3[] positions)
        {
            Debug.Log("GenerateStretch");
            int VertexAt(int x, int y)
            {
                return x * (_resolution + 1) + y;
            };
            float DistanceBetween(int idx1, int idx2)
            {
                return math.length(positions[idx1] - positions[idx2]);
            };

            for (int x = 0; x < _resolution + 1; x++)
            {
                for (int y = 0; y < _resolution + 1; y++)
                {
                    int idx1, idx2;

                    if (y != _resolution)
                    {
                        idx1 = VertexAt(x, y);
                        idx2 = VertexAt(x, y + 1);
                        m_solver.AddStretch(_indexOffset.x + idx1, _indexOffset.x + idx2, DistanceBetween(idx1, idx2));
                    }

                    if (x != _resolution)
                    {
                        idx1 = VertexAt(x, y);
                        idx2 = VertexAt(x + 1, y);
                        m_solver.AddStretch(_indexOffset.x + idx1, _indexOffset.x + idx2, DistanceBetween(idx1, idx2));
                    }

                    if (y != _resolution && x != _resolution)
                    {
                        idx1 = VertexAt(x, y);
                        idx2 = VertexAt(x + 1, y + 1);
                        m_solver.AddStretch(_indexOffset.x + idx1, _indexOffset.x + idx2, DistanceBetween(idx1, idx2));

                        idx1 = VertexAt(x, y + 1);
                        idx2 = VertexAt(x + 1, y);
                        m_solver.AddStretch(_indexOffset.x + idx1, _indexOffset.x + idx2, DistanceBetween(idx1, idx2));
                    }
                }
            }
            m_solver.UpdateStretch();
        }

        //void GenerateBending(int[] indices)
        //{
        //    // HACK: not for every kind of mesh
        //    for (int i = 0; i<indices.Length; i += 6)
        //    {
        //        int idx1 = indices[i];
        //        int idx2 = indices[i + 1];
        //        int idx3 = indices[i + 2];
        //        int idx4 = indices[i + 5];

        //        // TODO: calculate angle
        //        float angle = 0;
        //        m_solver.AddBend(m_indexOffset.x + idx1, m_indexOffset.x + idx2, m_indexOffset.x + idx3, m_indexOffset.x + idx4, angle);
        //    }
        //}

        void GenerateAttach(Vector3[] positions)
        {
            if (_attachedObjects != null)
            {
                int count = _attachedObjects.Length;
                _attachedIndices = new List<int>();
                for (int slotIdx = 0; slotIdx < count; slotIdx++)
                {
                    float distLast = float.MaxValue;
                    int nearest = 0;
                    for (int i = 0; i < positions.Length; i++)
                    {
                        float dist = math.distancesq(positions[i], _attachedObjects[slotIdx].position);
                        if (dist < distLast)
                        {
                            distLast = dist;
                            nearest = i;
                        }
                    }
                    _attachedIndices.Add(nearest);

                    int particleID = _attachedIndices[slotIdx];
                    Vector3 slotPos = positions[particleID];
                    m_solver.AddAttachSlot(slotPos, _attachedObjects[slotIdx]);
                    for (int i = 0; i < positions.Length; i++)
                    {
                        float restDistance = math.length(slotPos - positions[i]);
                        m_solver.AddAttach(_indexOffset.x + i, slotIdx, restDistance);
                    }
                    //m_solver.AddAttach(idx, positions[idx], 0);
                }

                m_solver.UpdateAttachSlot();
                m_solver.UpdateAttach();
            }
        }

        int slot = -1;
        public void AddAttach(Transform attachPoint)
        {
            var mesh = GetComponent<MeshFilter>().mesh;
            GPUClothDynamics.GetMesh(this, mesh, _meshProxy, false ? int2.zero : _indexOffset, _applyTransformAtExport, false, this.transform, _objBuffers, GetPositionsBuffer(), GetNormalsBuffer(), version: 2);
            Vector3[] attachPositions = mesh.vertices;
            float distLast = float.MaxValue;
            int nearest = 0;
            for (int i = 0; i < attachPositions.Length; i++)
            {
                float dist = math.distancesq(attachPositions[i], attachPoint.position);
                if (dist < distLast)
                {
                    distLast = dist;
                    nearest = i;
                }
            }

            Vector3 slotPos = attachPositions[nearest];
            m_solver.AddAttachSlot(slotPos, attachPoint);
            slot += 1;
            for (int i = 0; i < attachPositions.Length; i++)
            {
                float restDistance = math.length(slotPos - attachPositions[i]);
                if (restDistance < 0.03f)
                    m_solver.AddAttach(_indexOffset.x + i, slot, restDistance);
            }
            m_solver.UpdateAttachSlot();
            m_solver.UpdateAttach();
        }
        public void RemoveAllAttach()
        {
            slot = -1;
            m_solver.RemoveAllAttach();
            m_solver.UpdateAttachSlot();
            m_solver.UpdateAttach();
        }


        public void SetAttachedIndices(List<int> indices)
        {
            _attachedIndices = indices;
        }


        private void SetupSkinningHD(bool debugMode, Mesh mesh)
        {
            if (!_useMeshProxy || _meshProxy == null) return;

            Matrix4x4 initTransform = Matrix4x4.identity;// _meshProxy.transform.localToWorldMatrix;

            //var wpos = _positions;
            //for (int i = 0; i < _numParticles; i++)
            //{
            //	wpos[i] = transform.TransformPoint(wpos[i]);
            //}

            List<Vector3> positionVec = new List<Vector3>(mesh.vertices);
            List<int> tris = new List<int>(mesh.triangles);

            var dictTris = new Dictionary<Vector3, List<int>>();//dictTris.Clear ();
            for (int f = 0; f < tris.Count; f += 3)
            {
                if (dictTris.ContainsKey(positionVec[tris[f]]))
                {
                    var list = dictTris[positionVec[tris[f]]];
                    list.Add(tris[f + 1]);
                    list.Add(tris[f + 2]);
                }
                else
                {
                    dictTris.Add(positionVec[tris[f]], new List<int>(new[] {
                                            tris [f + 1],
                                            tris [f + 2]
                                        }));
                }
                if (dictTris.ContainsKey(positionVec[tris[f + 1]]))
                {
                    var list = dictTris[positionVec[tris[f + 1]]];
                    list.Add(tris[f + 2]);
                    list.Add(tris[f]);
                }
                else
                {
                    dictTris.Add(positionVec[tris[f + 1]], new List<int>(new[] {
                                            tris [f + 2],
                                            tris [f]
                                        }));
                }
                if (dictTris.ContainsKey(positionVec[tris[f + 2]]))
                {
                    var list = dictTris[positionVec[tris[f + 2]]];
                    list.Add(tris[f]);
                    list.Add(tris[f + 1]);
                }
                else
                {
                    dictTris.Add(positionVec[tris[f + 2]], new List<int>(new[] {
                                            tris [f],
                                            tris [f + 1]
                                        }));
                }
            }

            List<Vector2Int> connectionInfoTet = new List<Vector2Int>();
            List<int> connectedVertsTet = new List<int>();

            int maxVertexConnectionLow = 0;
            for (int n = 0; n < positionVec.Count; n++)
            {
                int start = connectedVertsTet.Count;
                if (dictTris.ContainsKey(positionVec[n]))
                {
                    var list = dictTris[positionVec[n]];
                    for (int i = 0; i < list.Count; i++)
                    {
                        connectedVertsTet.Add(list[i]);
                    }
                }
                int end = connectedVertsTet.Count;
                maxVertexConnectionLow = Mathf.Max(maxVertexConnectionLow, end - start);
                connectionInfoTet.Add(new Vector2Int(start, end));
            }

            if (debugMode) Debug.Log("<color=blue>CD: </color>positionVec: " + positionVec.Count);

            _connectionInfoTetBuffer = new ComputeBuffer(connectionInfoTet.Count, sizeof(int) * 2);
            _connectionInfoTetBuffer.SetData(connectionInfoTet.ToArray());
            _connectedVertsTetBuffer = new ComputeBuffer(connectedVertsTet.Count, sizeof(int));
            _connectedVertsTetBuffer.SetData(connectedVertsTet.ToArray());

            Mesh meshHD = _meshProxy.GetComponent<MeshFilter>()?.mesh;
            if (meshHD == null) { Debug.Log("<color=blue>CD: </color><color=orange>Missing mesh data in proxy object " + _meshProxy.name + "!</color>"); return; }

            SetSecondUVsForVertexID(meshHD);

            var name = this.transform.name + "_" + _meshProxy.name;
            var path = Path.Combine(Application.dataPath, "ClothDynamics/Resources/");

            var verts = meshHD.vertices;
            for (int i = 0; i < verts.Length; i++)
            {
                verts[i] = this.transform.InverseTransformPoint(_meshProxy.transform.TransformPoint(verts[i]));
            }
            GPUClothDynamics.FindControls(debugMode, this.name + _meshProxy.name, out _vertexBlendsBuffer, out _bonesStartMatrixBuffer, verts, meshHD.vertexCount, positionVec, tris, connectionInfoTet, connectedVertsTet, _weightsCurve._curve, 4, _weightsToleranceDistance, _scaleWeighting, initTransform, _skinPrefab, name, path, _minRadius);

            _startVertexBuffer = new ComputeBuffer(meshHD.vertexCount, sizeof(float) * 3);
            _startVertexBuffer.SetData(meshHD.vertices);

            //var smr = _meshProxy.GetComponent<SkinnedMeshRenderer>();
            //SetClothShader(smr);

            var mr = _meshProxy.GetComponent<MeshRenderer>();
            SetClothShader(mr);

            _mpbHD = new MaterialPropertyBlock();

            _csNormalsKernel = _clothSolver.FindKernel("CSNormals");
            _skinningHDKernel = _clothSolver.FindKernel("SkinningHD");

            int _workGroupSize = 256;
            int maxVerts = _objBuffers[1].positionsBuffer.count;
            _numGroups_VerticesHD = maxVerts.GetComputeShaderThreads(_workGroupSize);

            _mr.enabled = false;
        }

        internal void ComputeSkinningHD()
        {
            if (!_useMeshProxy || _meshProxy == null) return;

            //Debug.Log("ComputeSkinningHD");

            int maxVerts = _objBuffers[1].positionsBuffer.count;
            _clothSolver.SetInt(_maxVerts_ID, maxVerts);

            //_mpb.SetBuffer(_mpb_positionsBuffer_ID, /*m_solver._localSetup ? m_solver.positions2 :*/ m_solver._positions);

            //_clothSolver.SetBuffer(_skinningHDKernel, _positions_ID, _objBuffers[0].positionsBuffer);

            _clothSolver.SetMatrix(_localToWorldMatrix_ID, this.transform.localToWorldMatrix);
            _clothSolver.SetInt("_indexOffset", _indexOffset.x);
            _clothSolver.SetBuffer(_skinningHDKernel, _positions_ID, m_solver._positions);
            _clothSolver.SetBuffer(_skinningHDKernel, "_vertexBufferHD", _objBuffers[1].positionsBuffer);
            _clothSolver.SetBuffer(_skinningHDKernel, "_startVertexBuffer", _startVertexBuffer);
            _clothSolver.SetBuffer(_skinningHDKernel, "_bonesStartMatrixBuffer", _bonesStartMatrixBuffer);
            _clothSolver.SetBuffer(_skinningHDKernel, "_vertexBlendsBuffer", _vertexBlendsBuffer);
            _clothSolver.SetBuffer(_skinningHDKernel, "_connectionInfoTetBuffer", _connectionInfoTetBuffer);
            _clothSolver.SetBuffer(_skinningHDKernel, "_connectedVertsTetBuffer", _connectedVertsTetBuffer);

            _clothSolver.SetInt(_maxVerts_ID, maxVerts);
            _clothSolver.SetMatrix(_worldToLocalMatrixHD_ID, _meshProxy.transform.worldToLocalMatrix);//TODO set in skin shader?
            _clothSolver.Dispatch(_skinningHDKernel, _numGroups_VerticesHD, 1, 1);

            ComputeNormals(1);
            _mpbHD.SetBuffer(_mpb_normalsBuffer_ID, _objBuffers[1].normalsBuffer);
            _mpbHD.SetBuffer(_mpb_positionsBuffer_ID, _objBuffers[1].positionsBuffer);
            Renderer mr = _meshProxy.GetComponent<MeshRenderer>();
            //if(mr == null) mr = _meshProxy.GetComponent<SkinnedMeshRenderer>();
            if (mr != null) mr.SetPropertyBlock(_mpbHD);
        }
        private void ComputeNormals(int index)
        {
            _clothSolver.SetInt(_maxVerts_ID, _objBuffers[index].positionsBuffer.count);
            _clothSolver.SetBuffer(_csNormalsKernel, _positions_ID, _objBuffers[index].positionsBuffer);
            _clothSolver.SetBuffer(_csNormalsKernel, _connectionInfoBuffer_ID, _objBuffers[index].connectionInfoBuffer);
            _clothSolver.SetBuffer(_csNormalsKernel, _connectedVertsBuffer_ID, _objBuffers[index].connectedVertsBuffer);
            _clothSolver.SetBuffer(_csNormalsKernel, _normalsBuffer_ID, _objBuffers[index].normalsBuffer);
            _clothSolver.Dispatch(_csNormalsKernel, _objBuffers[index].positionsBuffer.count.GetComputeShaderThreads(256), 1, 1);
        }

        public void ExportMesh(bool useProxy = false)
        {
            var mesh = this.GetComponent<MeshFilter>() ? this.GetComponent<MeshFilter>().sharedMesh : this.GetComponent<SkinnedMeshRenderer>() ? this.GetComponent<SkinnedMeshRenderer>().sharedMesh : null;
            GPUClothDynamics.ExportMesh(this, mesh, _meshProxy, useProxy ? int2.zero : _indexOffset, _applyTransformAtExport, useProxy, this.transform, _objBuffers, GetPositionsBuffer(), GetNormalsBuffer(), version: 2);
        }

        public Mesh MyGetMesh()
        {
            var mesh = GetComponent<MeshFilter>().mesh;
            return GPUClothDynamics.GetMesh(this, mesh, _meshProxy, false ? int2.zero : _indexOffset, _applyTransformAtExport, false, this.transform, _objBuffers, GetPositionsBuffer(), GetNormalsBuffer(), version: 2);
        }
        public Mesh GetMesh(bool useProxy = false)
        {
            var mesh = this.GetComponent<MeshFilter>() ? this.GetComponent<MeshFilter>().sharedMesh : this.GetComponent<SkinnedMeshRenderer>() ? this.GetComponent<SkinnedMeshRenderer>().sharedMesh : null;
            return GPUClothDynamics.GetMesh(this, mesh, _meshProxy, useProxy ? int2.zero : _indexOffset, _applyTransformAtExport, useProxy, this.transform, _objBuffers, GetPositionsBuffer(), GetNormalsBuffer(), version: 2);
        }

        public void SetupFollowObjects(int[] indices = null)
        {
            if (_followObjects != null && _followObjects.Length > 0)
            {
                var buffer = GetPositionsBuffer();
                if (buffer != null)
                {
                    var data = new Vector3[buffer.count];
                    buffer.GetData(data);

                    if (_offsetPos == null) _offsetPos = new Vector3[_followObjects.Length];
                    if (_closestCenter == null) _closestCenter = new Vector3Int[_followObjects.Length];
                    if (_rightVec == null) _rightVec = new Vector3[_followObjects.Length];
                    if (_offsetRot == null) _offsetRot = new Quaternion[_followObjects.Length];

                    for (int i = 0; i < _followObjects.Length; i++)
                    {
                        if (indices != null)
                        {
                            int trisCount = indices.Length / 3;
                            float lastDist = float.MaxValue;
                            for (int n = 0; n < trisCount; n++)
                            {
                                var v0 = data[indices[n * 3 + 0]];
                                var v1 = data[indices[n * 3 + 1]];
                                var v2 = data[indices[n * 3 + 2]];
                                Vector3 center = this.transform.TransformPoint((v0 + v1 + v2) * 0.333f);

                                var dist = math.distance(center, _followObjects[i].transform.position);
                                if (dist < lastDist)
                                {
                                    lastDist = dist;
                                    _closestCenter[i].x = indices[n * 3 + 0];
                                    _closestCenter[i].y = indices[n * 3 + 1];
                                    _closestCenter[i].z = indices[n * 3 + 2];
                                }
                            }
                            var pos = this.transform.TransformPoint((data[_closestCenter[i].x] + data[_closestCenter[i].y] + data[_closestCenter[i].z]) * 0.333f);
                            _offsetPos[i] = _followObjects[i].transform.position - pos;
                            _offsetRot[i] = _followObjects[i].transform.localRotation;
                            _rightVec[i] = -_followObjects[i].transform.forward;
                        }
                        
                        Vector3 point1 = data[_closestCenter[i].x];
                        Vector3 point2 = data[_closestCenter[i].y];
                        Vector3 point3 = data[_closestCenter[i].z];

                        var posData = this.transform.TransformPoint((point1 + point2 + point3) * 0.333f);
                        _followObjects[i].transform.position = posData + _offsetPos[i];

                        // Calculate the normal vector of the triangle
                        Vector3 side1 = point2 - point1;
                        Vector3 side2 = point3 - point1;
                        Vector3 normal = Vector3.Cross(side1, side2).normalized;

                        float3 right = math.normalize(point3 * 100 - point1 * 100);

                        //TODO find better solution to keep z rotation
                        right = _rightVec[i];

                        normal = math.normalize(normal);
                        right = math.normalize(math.cross(normal, right));
                        float3 up = math.normalize(math.cross(right, normal));

                        float4x4 result = float4x4.identity;
                        result[0] = new float4(right.x, up.x, normal.x, 0);
                        result[1] = new float4(right.y, up.y, normal.y, 0);
                        result[2] = new float4(right.z, up.z, normal.z, 0);
                        result[3] = new float4(0, 0, 0, 1);

                        Matrix4x4 mat = Matrix4x4.identity;
                        mat.SetColumn(0, (Vector3)right);
                        mat.SetColumn(1, (Vector3)up);
                        mat.SetColumn(2, normal);
                        mat.SetColumn(3, new Vector4(0, 0, 0, 1));

                        // Calculate the relative rotation of the object
                        Quaternion relativeRotation = mat.ExtractRotation() * _offsetRot[i];// * Quaternion.Inverse(Quaternion.Euler(-90, 0, 0));

                        // Apply the relative rotation to the object
                        _followObjects[i].transform.rotation = relativeRotation;
                        
                        if(_originalParent != null && _dynamics._solver._localSpace)
                            _followObjects[i].transform.rotation = _originalParent.rotation * relativeRotation;

                    }
                }
            }

            //if (_followObjects != null)
            //{
            //    for (int i = 0; i < _followObjects.Length; i++)
            //    {
            //        var r = _followObjects[i].GetComponent<Renderer>();
            //        if (_mpb != null)
            //        {
            //            //Debug.Log("SetupFollowObjects " + _followObjects[i].name);
            //            r.SetPropertyBlock(_mpb);
            //        }
            //    }
            //}
        }

        Vector3[] _offsetPos;
        Quaternion[] _offsetRot;
        Vector3[] _rightVec;
        Vector3Int[] _closestCenter;
        //public Vector3 _originalEuler = Vector3.zero;

        float4 worldPosSkinning(float3 vertexPos, Vector3[] vertices, Vector3[] normals)
        {
            //int indexOffset = 0;

            float4x4 final = float4x4.identity;
            final[0] = new float4(0, 0, 0, 0);
            final[1] = new float4(0, 0, 0, 0);
            final[2] = new float4(0, 0, 0, 0);
            final[3] = new float4(0, 0, 0, 0);

            for (int n = 0; n < 1; n++)
            {
                int num = n;// uint(_vertexBlendsBuffer[index].bones[n]);

                //int start = _connectionInfoTetBuffer[num].x;
                //int end = _connectionInfoTetBuffer[num].y - 1;
                float3 normal = normals[n];// new float3(0, 0, 0);

                float3 pos0 = vertices[n];// mul(_localToWorldMatrix, float4(_positions[indexOffset + num].xyz, 1)).xyz;

                //for (uint j = start; j < end; j += 2)
                //{
                //    int indexA = _connectedVertsTetBuffer[j];
                //    int indexB = _connectedVertsTetBuffer[j + 1];
                //    float3 a = mul(_localToWorldMatrix, float4(_positions[indexOffset + indexA].xyz, 1)).xyz * 100 - pos0 * 100;
                //    float3 b = mul(_localToWorldMatrix, float4(_positions[indexOffset + indexB].xyz, 1)).xyz * 100 - pos0 * 100;
                //    normal += cross(a, b);
                //}

                float3 right = math.normalize(((float3)vertices[(n + 1) % 3]).xyz * 100 - pos0 * 100);
                normal = math.normalize(normal);
                right = math.normalize(math.cross(normal, right));
                float3 up = math.normalize(math.cross(right, normal));

                float4x4 result = float4x4.identity;
                result[0] = new float4(right.x, up.x, normal.x, pos0.x);
                result[1] = new float4(right.y, up.y, normal.y, pos0.y);
                result[2] = new float4(right.z, up.z, normal.z, pos0.z);
                result[3] = new float4(0, 0, 0, 1);

                final = result;// mul(result, _bonesStartMatrixBuffer[num]);
            }
            return math.mul(final, new float4(vertexPos.xyz, 1));//mul (final, float4(vertexPos.xyz,1));//
        }

        private void OnDrawGizmos()
        {

        }


        private void OnDestroy()
        {
            if (_objBuffers != null)
                for (int i = 0; i < _objBuffers.Length; i++)
                {
                    if (_objBuffers[i] != null)
                    {
                        _objBuffers[i].positionsBuffer.ClearBuffer();
                        _objBuffers[i].normalsBuffer.ClearBuffer();
                        _objBuffers[i].connectionInfoBuffer.ClearBuffer();
                        _objBuffers[i].connectedVertsBuffer.ClearBuffer();
                    }
                }
            _connectionInfoTetBuffer.ClearBuffer();
            _connectedVertsTetBuffer.ClearBuffer();
            _startVertexBuffer.ClearBuffer();
            _vertexBlendsBuffer.ClearBuffer();
            _bonesStartMatrixBuffer.ClearBuffer();
            _tempBuffer.ClearBuffer();
        }

        private void OnEnable()
        {
#if UNITY_EDITOR
            EditorApplication.playModeStateChanged += OnPlaymodeChanged;
#endif
        }

        private void OnDisable()
        {
#if UNITY_EDITOR
            EditorApplication.playModeStateChanged -= OnPlaymodeChanged;
#endif
        }

#if UNITY_EDITOR
        public void OnPlaymodeChanged(PlayModeStateChange state)
        {
            if (state == PlayModeStateChange.EnteredPlayMode)
            {
                //PlayerPrefs.SetFloat(this.GetInstanceID().ToString(), this._bufferScale);
                //PlayerPrefs.Save();
            }
            else if (state != PlayModeStateChange.EnteredPlayMode && !EditorApplication.isPlayingOrWillChangePlaymode)
            {
                //this._bufferScale = math.ceil(PlayerPrefs.GetFloat(this.GetInstanceID().ToString()));

                if (_meshProxy != null)
                {
                    var skinFileName = PlayerPrefs.GetString("CD_" + this.name + _meshProxy.name + "skinPrefab", "");
                    Debug.Log("skinFileName " + skinFileName);
                    if (!string.IsNullOrEmpty(skinFileName))
                    {
                        TextAsset newPrefabAsset = Resources.Load(skinFileName) as TextAsset;
                        _skinPrefab = newPrefabAsset;
                        Debug.Log("_skinPrefab " + _skinPrefab.name);
                    }
                }
                //if (_usePreCache)
                //{
                //    var cacheFileName = PlayerPrefs.GetString("CD_" + this.name + "clothPrefab", "");
                //    if (!string.IsNullOrEmpty(cacheFileName))
                //    {
                //        TextAsset newPrefabAsset = Resources.Load(cacheFileName) as TextAsset;
                //        _preCacheFile = newPrefabAsset;
                //    }
                //}
            }
        }
#endif

    }
}